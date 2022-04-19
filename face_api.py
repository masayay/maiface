from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from PIL import Image
from face_detect import FaceDetector
from logging import getLogger
from utils import face_api_response as far
from utils.face_api_response import Group, GROUPS, ApiResponse, Face
from typing import List
import io
import conf
from sqlalchemy.orm import Session
#import utils.db_util as db_util
"""
Load Configuration
"""
# Logger
logger = getLogger(conf.LOG_OUTPUT)
logger.setLevel(conf.LOG_LEVEL)
# API
app = FastAPI(title=conf.API_TITLE,
              version=conf.API_VERSION)
# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount templates
templates = Jinja2Templates(directory="templates")
# Load detector
detector = FaceDetector()

"""
page
"""
@app.get("/")
async def recognition(request: Request):
    """
    Face recognition page
    """
    return templates.TemplateResponse("recognition.html", {"request": request})

@app.get("/add")
async def registration(request: Request):
    """
    Face registration page
    """
    return templates.TemplateResponse("registration.html", {"request": request,
                                                            "groups": GROUPS})

"""
api
"""
@app.get("/api/groups", response_model=List[Group])
async def get_groups():
    return GROUPS

@app.get("/api/faceid_infos", response_model=List[Face])
async def get_faceid_infos():
    faces = detector.get_faces()
    return faces

@app.get("/api/faceid_info/{faceid}", response_model=Face)
async def get_faceid_info(faceid: str):
    face = detector.get_face(faceid)
    return face

@app.post("/api/face_recognizer")
async def api_face_recognizer(deviceid: str = Form(""),
                              image: UploadFile = File(...)):
    """
    Parameters
    ----------
    image : Binary

    Returns
    -------
    JSONResponse
    """
    try:
        # Get the image
        image_data = await image.read()
        # Change image type from byteio to PIL
        img_pil = Image.open(io.BytesIO(image_data))
        # Serch faceid
        results = detector.face_search(img_pil, deviceid)
        # Convert to JSON
        json_response = far.create_face_response(results)
        
        return JSONResponse(content=json_response)
 
    except Exception as e:
        logger.error('POST /api/face_recognizer: %e' % e)
        return e

@app.post("/api/face_register", response_model=ApiResponse)
async def api_face_register(userid: str = Form(...),
                            groupid: str = Form(""),
                            image: UploadFile = File(...)):
    """
    Create faceid.
    
    Parameters
    ----------
    userid : str,
    groupid : str,
    image : Binary

    Returns
    -------
    JSONResponse
    """
    try:
        # Create response
        res = ApiResponse()
        res.groupid = groupid
        res.userid = userid
        faceid = groupid + "-" + userid
        res.faceid = faceid
        
        # Check Argument
        if not far.isonlynum(userid):
            return far.create_response(res, "E2001")
        if not far.isonlynum(groupid):
            return far.create_response(res, "E2002")
        
        # Get image
        image_data = await image.read()
        if not image_data:
            return far.create_response(res, "E2004")
        # Change image type from byteio to PIL
        img_pil = Image.open(io.BytesIO(image_data))

        # Create, Save and add embedding
        r_code = detector.register_faceid(faceid, img_pil)
        
        # Return response
        if r_code == 1:
            return far.create_response(res, "S1001")
        else:
            return far.create_response(res, "E1002")
    
    except Exception as e:
        logger.error('POST /api/face_register: %e' % e)
        return far.create_response(res, "E1001")

@app.delete("/api/{groupid}/{userid}", response_model=ApiResponse) 
async def api_delete_faceid(groupid: str, userid: str):
    """
    Delete faceid
    """
    try:
        # Create response
        res = ApiResponse()
        res.groupid = groupid
        res.userid = userid
        faceid = groupid + "-" + userid
        res.faceid = faceid
        
        # Check Argument
        if not far.isonlynum(userid):
            return far.create_response(res, "E2001")
        if not far.isonlynum(groupid):
            return far.create_response(res, "E2002")

        # Delete faceid
        r_code = detector.delete_faceid(faceid)
        
        # Return Response
        if r_code == 1:
            return far.create_response(res, "S1011")
        else:
            return far.create_response(res, "E1011")
            
    except Exception as e:
        logger.error('DELETE /api/{groupid}/{userid}: %e' % e)
        return e
