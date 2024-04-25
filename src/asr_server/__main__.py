from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil, glob, os
from typing import List
import uvicorn

import ffmpeg

import whisperx
import subprocess

origins = ["*"]

device = "cpu"
batch_size = 1
compute_type = "int8"

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

app = FastAPI(title="Upload Files by FastAPI")

@app.post("/upload", tags=['Upload files to directory'])
async def upload_files(files: List[UploadFile] = File(...)):
    file_list = []
    for single_file in files:
        with open(f'uploaded_files/{single_file.filename}', "wb+") as file_object:
            shutil.copyfileobj(single_file.file, file_object)
        file_list.append(single_file.filename)

    return "The following files has been uploaded:" + str(file_list)

@app.post("/transcribe_files", tags=['Transcribe audio files'])
async def transcribe_audio_files(files: List[UploadFile] = File(...)):
    for file in files:
        filename = f"uploaded_files/{file.filename}"
        with open(filename, "wb") as f:
            f.write(await file.read()) # Save the uploaded file to disk

            audio = whisperx.load_audio(filename)
            result = model.transcribe(audio, batch_size=batch_size)
            if "segments" in result:
                # transcription = result["segments"][0]["text"]
                transcription = [result["segments"][i]["text"] for i in range(len(result["segments"]))]
                # transcription = result
            else:
                transcription = None

    # print("Transcription:" + transcription)
    # return "Transcription:" + transcription
    print(transcription)
    return transcription


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def main():
    uvicorn.run(app, host="0.0.0.0")

if __name__ == "__main__": main()
