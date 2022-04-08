// Capture camera image
function copyFrame() {
    // Create canvas
    var canvas_capture_image = document.createElement('canvas');
    canvas_capture_image.id = "captureImage"; 
    var va = document.getElementById("myVideo");
    let parentDiv = va.parentNode
    parentDiv.insertBefore(canvas_capture_image, va)
    var cci = canvas_capture_image.getContext("2d");

    canvas_capture_image.width  = va.videoWidth;
    canvas_capture_image.height = va.videoHeight;
    cci.drawImage(va, 0, 0);
}

// Delete capture image
function delFrame() {

    var canvas_capture_image = document.getElementById("captureImage");
    canvas_capture_image.parentNode.removeChild(canvas_capture_image);
        
}

// canvas image to blob
function canvasToBlob(canvas, callback, type) {
    if (!type) {
        type = 'image/jpeg';
    }
    if (canvas.toBlob) {
        canvas.toBlob(callback, type);
    } else if (canvas.toDataURL && window.Uint8Array && window.Blob && window.atob) {
        var binStr = atob(canvas.toDataURL(type).replace(/^[^,]*,/, '')),
        len = binStr.length,
        arr = new Uint8Array(len);

        for (var i = 0; i < len; i++) {
            arr[i] = binStr.charCodeAt(i);
        }

        callback(new Blob([arr], { type: type }));
    } else {
        callback(null);
    }
}

/**
 * Add canvas image to FormData and post
 * @param {Element} formElem : Form DOM node
 */
function send(formElem) {
    var canvasElem = document.getElementById('captureImage');

    if (window.FormData) {
        canvasToBlob(canvasElem, function (canvasBlob) {
            // If get canvas image as blob object, add image to FormData
            if (canvasBlob) {
                // Create FormData object
                var fd = new FormData(formElem);
                // Add canbas Blob image to FormData
                fd.append('image', canvasBlob);
                // Create XMLHttpRequest object
                var xhr = new XMLHttpRequest();
                // Connect to API Server
                xhr.open('POST', formElem.action, true);
                xhr.onload = function () {
                    if (this.status === 200) {
                        let objects = JSON.parse(this.response)
                        console.log(objects);
                        alert(objects.msg)
                    } else {
                        console.error(xhr);
                    }
                };
                //Send FormData object 
                xhr.send(fd);
                // Dlete capture image
                delFrame()
                // clear formdata
                var userid = document.getElementById('userid');
                userid.value = '';
                var groupSelect = document.getElementById('groupid');
                groupSelect.selectedIndex = 0
            } else {
                alert('Cannot send image.\nBrowser cannot create blob object.');
            }
        }, 'image/jpeg');
    } else {
        alert('Cannot send image.\nBrowser cannot create FormData objext.');
    }
}

document.getElementById('faceid_regist_form').addEventListener('submit', function (e) {
    const userid = document.getElementById('userid').value;
    const captureImage = document.getElementById('captureImage');
    let groupSelect = document.getElementById('groupid');
    const groupid = groupSelect.options[groupSelect.selectedIndex].value;
    
    if(!userid) {
        alert("Enter user id");
        event.stopPropagation(); 
        event.preventDefault();
    } else if (!groupid) {
        alert("Chose group id");
        event.stopPropagation();
        event.preventDefault();
    } else if (!captureImage) {
        alert("Capture image.");
        event.stopPropagation();
        event.preventDefault();
    } else {
        var formElem = this;
        e.preventDefault();
        send(formElem);
    }
}, false);

