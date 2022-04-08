from fastapi.encoders import jsonable_encoder
from typing import Optional
from pydantic import BaseModel
import re

"""
Group
"""
class Group(BaseModel):
    groupid: Optional[str]
    groupname: Optional[str]

GROUPS = [
    {"groupid":"100", "groupname":"Group1"},
    {"groupid":"999", "groupname":"Others"},
]

def get_group_name(gid):
    gname = next((x for x in GROUPS if x["groupid"] == gid), None)["groupname"]
    return gname

"""
User
"""
class Face(BaseModel):
    faceid: Optional[str]
    groupid: Optional[str]
    userid: Optional[str]
    numfiles: Optional[int]



"""
Response Message
"""
class Message(BaseModel):
    mid: Optional[str]
    msg: Optional[str]

MESSAGES = [
    {"mid":"S1001", "msg":"Registered"},
    {"mid":"E1001", "msg":"Cannot register"},
    {"mid":"E1002", "msg":"Already registered"},
    {"mid":"S1011", "msg":"Deleted"},
    {"mid":"E1011", "msg":"Cannot delete"},
    {"mid":"E2001", "msg":"userid only accept numbers"},
    {"mid":"E2002", "msg":"groupid  only accept numbers"},
    {"mid":"E2003", "msg":"only accept 0-9,a-z,A-Z"},
    {"mid":"E2004", "msg":"Cannot get image"},
]

def get_message(mid):
    msg = next((x for x in MESSAGES if x["mid"] == mid), None)["msg"]
    return msg

"""
Api Reponse based on message
"""
class ApiResponse(Message):
    result: str = 'SUCCESS'
    groupid: Optional[str]
    userid: Optional[str]
    faceid: Optional[str]

def create_response(response, mid):
    """
    Create ApiResponse

    Parameters
    ----------
    response : ApiResponse
    mid : str
    
    Returns
    -------
    response : ApiResponse
    """
    response.mid = mid
    if mid[0] == "S":
        response.result = "SUCCESS"
    else:
        response.result = "ERROR"
    response.msg = get_message(mid)
    
    return response

"""
Face detection response
"""
class FaceDetectResponse(BaseModel):
    result: str = 'no results'
    faceid: Optional[str]
    userid: Optional[str]
    username: Optional[str]
    groupid: Optional[str]
    groupname: Optional[str]
    probability: Optional[float]
    x: Optional[float]
    y: Optional[float]
    height: Optional[float]
    width: Optional[float]
    
def create_face_response(results):
    """
    Convert dictonary to json response

    Parameters
    ----------
    results : Dictionary
        [{'box':[x1,y1,x2,y2], 'id':XXXX, 'probability':99.9}, ...]
    Returns
    -------
    JSON
        {{"result": "detext", "faceid": "XXX-XXXX", "probability": "xxx",
          "x":, "y":, "height":, "width": }, ...}
    """
    responses = []
    if results is None:
        res = FaceDetectResponse()
        responses.append(res)
    else:
        for result in results:
            res = FaceDetectResponse()
            res.result = 'detect'
            res.faceid = result['id']
            if (res.faceid):
                try:
                    res.groupid = result['id'].split('-')[0]
                    res.userid = result['id'].split('-')[1]
                except IndexError:
                    res.userid = res.faceid
                try:
                    res.username = ''
                except KeyError:
                    res.username = res.faceid
                try:
                    res.groupname = get_group_name(res.groupid)
                except KeyError:
                    pass

            res.probability = result['probability']
            box = result['box']
            res.x = float(box[0])
            res.y = float(box[1])
            res.height = float(box[3] - box[1])
            res.width = float(box[2] - box[0])
            responses.append(res)
    
    return jsonable_encoder(responses)

"""
Input form Check
"""
def isonlystringnum(s):
    """
    Check string format with [a-zA-Z0-9_]
    Parameters
    ----------
    s : str

    Returns
    -------
    boolen
    """
    try:
        if re.fullmatch('^wd+$', s):
            result = True
        else:
            result = False
    except TypeError:
        result = False
        
    return result

def isonlynum(s):
    """
    Check string format with [0-9]
    Parameters
    ----------
    s : int

    Returns
    -------
    boolen
    """
    try:
        if re.fullmatch('^\d+$', s):
            result = True
        else:
            result = False
    except TypeError:
        result = False
        
    return result
