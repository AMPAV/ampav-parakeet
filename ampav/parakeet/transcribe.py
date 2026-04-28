import torch
from pathlib import Path
from ampav.core.schema import ToolOutput, Transcript, WordSegment, ParagraphSegment, AVMetadata
import time
import logging
import argparse
import nemo.collections.asr as nemo_asr
from ampav.core.media import ChunkedAudio
from ampav.core.logging import LOG_FORMAT
from ampav.core.file_formats.webvtt import paragraphs_to_webvtt
from ampav.core.gpu import ForceComputeDevice
import os

def transcribe_file(audiofile: Path, modelname: str="nvidia/parakeet-tdt-0.6b-v3", 
                    device: str | None=None,
                    chunk_size: int=30, chunk_overlap: int=5) -> ToolOutput:
    """Transcribe a file using parakeet"""
    
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
            for offsets, samples in ca.get_chunks(chunk_size, chunk_overlap=chunk_overlap):
                start_timestamp = offsets[0] 
                overlap_length = offsets[1]     
                h = model.transcribe([samples], return_hypotheses=True, timestamps=True,
                                    verbose=False)[0]
                for word in h.timestamp['word']:
                    words.append(WordSegment.from_str(word['word'], 
                                                      start_time=float(word['start'] + (start_timestamp - overlap_length)),
                                                      end_time=float(word['end'] + (start_timestamp - overlap_length)),
                                                      )) 
                
    # get the duration of the media file.
    av = AVMetadata.from_file(audiofile)
    xscript = Transcript(words= words,                         
                         media_duration=av.duration)
    xscript.remove_overlapping_words(separator=' ')
    
    logging.info("Transcription complete")
    output.output = xscript
    output.end_time = time.time()
    return output


def cli_parakeet_transcribe():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to transcribe using whisper")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-0.6b-v3", help="Model to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--debug", action="store_true", help="Enable debugging")
    parser.add_argument("--chunk_size", type=int, default=30, help="Size of chunks to process")
    parser.add_argument("--chunk_overlap", type=int, default=5, help="Number of seconds of audio overlap")
    parser.add_argument("--webvtt", action="store_true", help="Output webvtt instead of yaml")
    args = parser.parse_args()

    # NeMo logs like crazy, and I really don't want to see it on the console if I can
    # avoid it.    
    loggers = [logging.getLogger(name).name for name in logging.root.manager.loggerDict]
    for n in [x for x in loggers if x.startswith('nv') or x.startswith('nemo')]:
        logging.getLogger(n).setLevel(logging.ERROR)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
    
    xscript = transcribe_file(args.file, modelname=args.model, 
                              device=args.device,
                              chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if args.webvtt:
        print(paragraphs_to_webvtt(xscript.output.paragraphs))
    else:
        print(xscript.model_dump_yaml())
   
if __name__ == "__main__":
    cli_parakeet_transcribe()