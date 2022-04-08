// Parameters
const s = document.getElementById('objDetect');
const sourceVideo = s.getAttribute("data-source");  //the source video to use
const uploadWidth = s.getAttribute("data-uploadWidth") || 640; //the width of the upload file *shrink from orijginal size
const mirror = s.getAttribute("data-mirror") || false;  //mirror the boundary boxes
const apiServer = s.getAttribute("data-apiServer") || window.location.origin + '/api/face_recognizer'; //Face Detection API server url

// Video element selector
v = document.getElementById(sourceVideo);

// for starting events
let isPlaying = false,
    gotMetadata = false;

// Canvas setup
// create a canvas to grab an "image" for upload
// for uploading "Object Detection API"
let imageCanvas = document.createElement('canvas');
let imageCtx = imageCanvas.getContext("2d");
 
// create a canvas for "drawing object boundaries"
// for displaying our "boxes" and "labels"
let drawCanvas = document.createElement('canvas');
// document.body.appendChild(drawCanvas);
let parentDiv = v.parentNode
parentDiv.insertBefore(drawCanvas, v)
let drawCtx = drawCanvas.getContext("2d");

//initializing the deviceId with null
let deviceId = null;
  
//returns a promise that resolves to a list of
//connected devices
navigator.mediaDevices.enumerateDevices()
.then( devices => {
    devices.forEach(device => {
        //if the device is a video input
        //then get its deviceId
        if(device.kind === 'videoinput'){
            deviceId = device.deviceId;
        }
    });
})
.catch(err => {
    //handle the error
});

// draw boxes and labels on each detected object
function drawBoxes(objects) {

    // clear the previous drawings
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    //console.log(objects)

    // filter out objects that contain a class_name and then draw boxes and labels on each
    objects.filter(object => object.result).forEach(object => {

        let source_upload_ratio = uploadWidth / v.videoWidth
        let x = object.x / source_upload_ratio;
        let y = object.y / source_upload_ratio;
        let width = object.width / source_upload_ratio;
        let height = object.height / source_upload_ratio;
        let probability = Math.round(object.probability *10) / 10

        //flip the x axis if local video is mirrored
        if (mirror) {
            x = drawCanvas.width - (x + width);
        }
        if(object.groupname){
            drawCtx.fillText(object.groupname, x + 5, y + 20);
        }
        if(object.userid){
            drawCtx.fillText(object.userid, x + 5, y + 40);
        }
        drawCtx.fillText(probability + "%", x + 5, y + height - 5);
        drawCtx.strokeRect(x, y, width, height);

    });
}

function drawPeformance(result) {
    drawCtx.fillText(result + "msec", 5, drawCanvas.height - 10);
}

// Add file blob to a form and post
function postFile(file) {
    // Performance Measurement checkpoint
    const startTime = performance.now();
    
    //Set options as form data
    let formdata = new FormData();
    formdata.append("image", file);
    
    //console.log(deviceId);
    formdata.append("deviceid", deviceId);
    
    let xhr = new XMLHttpRequest();
    
    // async request
    xhr.open('POST', apiServer, true);
    xhr.onload = function () {
        // Performance Measurement checkpoint
        const endTime = performance.now();
        
        if (this.status === 200) {
            let objects = JSON.parse(this.response);
 
            // draw the boxes
            drawBoxes(objects);
            
            // draw performance
            result = Math.round(endTime - startTime);
            drawPeformance(result);
            
            // Send the next image
            imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
            imageCanvas.toBlob(postFile, 'image/jpeg');
        }
        else{
            console.error(xhr);
            // When get error, retry
            setTimeout(()=>{
                imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
                imageCanvas.toBlob(postFile, 'image/jpeg');
            },1000);
        }
    };
    
    xhr.send(formdata);
    
    // When get error, retry
    xhr.onerror = function(e){
        setTimeout(()=>{
            imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
            imageCanvas.toBlob(postFile, 'image/jpeg');
        },1000);
    };
}

// Start object detection
function startObjectDetection() {
 
    console.log("starting object detection");
 
    // Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;
 
    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);
 
    // Some styles for the drawcanvas
    drawCtx.lineWidth = "4";
    drawCtx.strokeStyle = "cyan";
    drawCtx.font = "20px Verdana";
    drawCtx.fillStyle = "cyan";

    // Save and send the first image, change image size
    imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
    // Change to Blob type
    imageCanvas.toBlob(postFile, 'image/jpeg');
 
}

// Starting events
// check if metadata is ready - we need the video size
v.onloadedmetadata = () => {
    console.log("video metadata ready");
    gotMetadata = true;
    if (isPlaying)
        startObjectDetection();
};

// see if the video has started playing
v.onplaying = () => {
    console.log("video playing");
    isPlaying = true;
    if (gotMetadata) {
        startObjectDetection();
    }
};



