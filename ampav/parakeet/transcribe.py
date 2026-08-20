import torch
from pathlib import Path
from ampav.core.schema import ToolOutput, Transcript, WordSegment, AVMetadata
import time
import logging
import argparse
import nemo.collections.asr as nemo_asr
from ampav.core.media import ChunkedAudio
from ampav.core.logging import LOG_FORMAT, ListLoggingHandler
from ampav.core.gpu import ForceComputeDevice
from ampav.core.utils import dump_data

def transcribe_file(audiofile: Path, modelname: str="nvidia/parakeet-tdt-0.6b-v3", 
                    device: str | None=None,
                    chunk_size: int=60, chunk_overlap: int=0) -> ToolOutput:
    """Create a transcript using parakeet

    Args:
        audiofile (Path): Audio file to transcribe
        modelname (str, optional): parakeet model to use. Defaults to "nvidia/parakeet-tdt-0.6b-v3".
        device (str | None, optional): compute device to use - cpu or gpu card name. Defaults to best available.
        chunk_size (int, optional): Size of audio chunks to process. Defaults to 60.
        chunk_overlap (int, optional): How much each chunk should overlap. Defaults to 0.

    Returns:
        ToolOutput: Transcription output
    """
    
    # create our output structure
    output = ToolOutput(tool_name="parakeet",                        
                        parameters={"model": modelname,
                                    "device": device,
                                    "content_source": str(audiofile),                                    
                                    },
                        start_time=time.time())
    
    # set the logging to log into our output structure
    output.setup_logging()

        # get the device if we need to
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Detected device {device}")
        output.parameters['device'] = device

    with ForceComputeDevice(device):
        logging.info(f"Using {device} for transcribing")
        model: nemo_asr.models.ASRModel = nemo_asr.models.ASRModel.from_pretrained(modelname)        
        words = []    
        logging.info(f"Chunking {audiofile} in {chunk_size}s chunks with an overlap of {chunk_overlap}s")
        with ChunkedAudio(audiofile, 0, sample_rate=16000, channels=1) as ca:        
            for start_timestamp, samples in ca.get_chunks(chunk_size, chunk_overlap=chunk_overlap):
                logging.debug(f"Processing chunk starting at: {start_timestamp}, sample size: {len(samples)}")
                h = model.transcribe([samples], return_hypotheses=True, timestamps=True,
                                    verbose=False)[0]
                for word in h.timestamp['word']:
                    words.append(WordSegment.from_str(word['word'], 
                                                      start_time=float(word['start'] + start_timestamp),
                                                      end_time=float(word['end'] + start_timestamp),
                                                      )) 
                    #logging.debug(f"{words[-1]}")
                
    # get the duration of the media file.
    av = AVMetadata.from_file(audiofile)
    xscript = Transcript(words= words,                         
                         media_duration=av.duration)
    xscript.remove_overlapping_words()
    logging.info(f"Finished transcript, {len(xscript.paragraphs)} paragraphs, {len(xscript.words)} words.")
    output.output = xscript
    output.end_time = time.time()
    return output


def cli_parakeet_transcribe():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path, help="File to transcribe using parakeet")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-0.6b-v3", help="Model to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("--chunk_size", type=int, default=60, help="Size of chunks to process")
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Number of seconds of audio overlap")
    parser.add_argument("--format", choices=['yaml', 'json', 'pickle'], default='yaml', help="Output format, default yaml")
    args = parser.parse_args()

    # NeMo logs like crazy, and I really don't want to see it on the console if I can
    # avoid it.    
    loggers = [logging.getLogger(name).name for name in logging.root.manager.loggerDict]
    for n in [x for x in loggers if x.startswith('nv') or x.startswith('nemo')]:
        logging.getLogger(n).setLevel(logging.ERROR)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    # capture the logging
    logs = []
    loghandler = ListLoggingHandler(logs)
    logging.getLogger().addHandler(loghandler)
            
    logging.info("Starting processing")
    start = time.time()    
    result = transcribe_file(args.file, modelname=args.model, 
                             device=args.device,
                             chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # update the tool_output structure with the runtime things            
    result.start_time = start
    result.end_time = time.time()            
    logging.info(f"Saving data to {args.output} in {args.format} format")
    result.messages = logs
    dump_data(result, args.format, args.output)

    
if __name__ == "__main__":
    cli_parakeet_transcribe()