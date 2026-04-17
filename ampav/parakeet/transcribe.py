import torch
from pathlib import Path
from ampav.core.schema import ToolOutput, Transcript, WordSegment, ParagraphSegment, AVMetadata
import time
import logging
import argparse
import nemo.collections.asr as nemo_asr
from ampav.core.media import ChunkedAudio
from ampav.core.formats.transcript.utils import words_to_paragraphs
from ampav.core.logging import LOG_FORMAT
from ampav.core.formats.transcript.webvtt import paragraphs_to_webvtt
import os

def transcribe_file(audiofile: Path, modelname: str="nvidia/parakeet-tdt-0.6b-v3", 
                    cpu_only: bool=False,
                    chunk_size: int=30, chunk_overlap: int=5) -> ToolOutput:
    """Transcribe a file using parakeet"""
    
    # create our output structure
    output = ToolOutput(tool_name="parakeet",                        
                        parameters={"model": modelname,
                                    "device": None,
                                    "content_source": str(audiofile),                                    
                                    },
                        start_time=time.time())
    
    # set the logging to log into our output structure
    output.setup_logging()

    if cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'        
    else:
        device = 'gpu'
    output.parameters['device'] = device
    logging.info(f"Using {device} for transcribing")
    model: nemo_asr.models.ASRModel = nemo_asr.models.ASRModel.from_pretrained(modelname)        
    
    words = []    
    logging.info(f"Chunking {audiofile} in {chunk_size}s chunks with an overlap of {chunk_overlap}s")
    with ChunkedAudio(audiofile, 0, sample_rate=16000, channels=1) as ca:        
        for position, overlap_offset, chunk_duration, samples in ca.get_chunk(chunk_size, chunk_overlap=chunk_overlap):
            h = model.transcribe([samples], return_hypotheses=True, timestamps=True,
                                 verbose=False)[0]
            # text stitching
            offset = position - overlap_offset
            new_words = [(x['word'], x['start'] + offset, x['end'] + offset) for x in h.timestamp['word']]
            while words and words[-1][1] > new_words[0][1]:
                words.pop()
            words.extend(new_words)
            
    # get the duration of the media file.
    av = AVMetadata.from_file(audiofile)
    xscript = Transcript(text=" ".join([x[0] for x in words]),
                         media_duration=av.duration)
    for w in words:
        xscript.words.append(WordSegment.from_str(w[0],
                                                  start_time=w[1],
                                                  end_time=w[2]))
    # we don't have the paragraphs, so we should synthesize them.
    xscript.paragraphs = words_to_paragraphs(xscript.words)
    
    logging.info("Transcription complete")
    output.output = xscript
    output.end_time = time.time()
    return output


def cli_parakeet_transcribe():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to transcribe using whisper")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-0.6b-v3", help="Model to use")
    parser.add_argument("--cpu_only", action="store_true", help="Only use CPU")
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
                              cpu_only=args.cpu_only,
                              chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if args.webvtt:
        print(paragraphs_to_webvtt(xscript.output.paragraphs))
    else:
        print(xscript.model_dump_yaml())
   
