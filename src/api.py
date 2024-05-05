from inference import TableExtractionPipeline
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, Body, File, Depends, Form
from pydantic import BaseModel, model_validator
from typing import List
from pdf2image import convert_from_bytes
import numpy as np
from io import BytesIO
import json
import logging
logging.basicConfig(filename='../../logs/fastapi_app.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


app = FastAPI()
logging.info("Initialised FastAPI interface.")

pipe = TableExtractionPipeline(str_config_path='structure_config.json', str_model_path='../../model.pth', str_device='cuda')
logging.info("TSR pipeline initialised.")


@app.get("/")
async def return_cells(message: str = Form(...), files: List[UploadFile]=File(...)):
    file_dict = {}
    message = json.loads(message)
    for f in files:
        file_dict[f.filename] = BytesIO(await f.read())
    
    logging.info("GET request recieved at the endpoint")
    response = []
    for bbox_result in message["bbox_result"]:
        pdf_path = bbox_result["pdf_file"]

        images = convert_from_bytes(file_dict[pdf_path].read(), dpi=300)
        img = images[bbox_result["page_num"]-1].convert('RGB')
        logging.info("PDF file loaded")
        print(img.size)
        for i, bbox in enumerate(bbox_result["bbox"]):
            img_cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            draw = ImageDraw.Draw(img_cropped)
            extracted_table = pipe.recognize(img_cropped, f"table_{bbox_result['page_num']}_{i}", [], out_cells=True)
            for cell in extracted_table["cells"][0]:
                bbx = cell["bbox"]
                print(cell)
                draw.rectangle(bbx, outline="red", width=10)
            
            #extracted_table = pipe.recognize(img_cropped, f"table_{bbox_result['page_num']}_{i}", [], out_cells=True)
            #for cell in extracted_table["cells"][0]:
            #    bbx = cell["bbox"]
            #    draw.rectangle(bbx, outline="blue", width=10)
            img_cropped.save("cropped.png")
            response.append(extracted_table["cells"])
            logging.info(f"table_{bbox_result['page_num']}_{i} extracted successfully")
    return response
