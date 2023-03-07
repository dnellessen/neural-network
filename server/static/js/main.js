const canvas = document.getElementById('drawing-area');
const ctx = canvas.getContext('2d');

const clearButton = document.getElementById('clear-button');
const detailsButton = document.getElementById("details-button");
const collapsible = document.getElementById("collapsible-container");
const details = document.getElementById('details')

const state = {
    mousedown: false,
};

ctx.lineWidth = 25;
ctx.lineCap = "round";
ctx.imageSmoothingEnabled = true;

canvas.addEventListener('mousedown', handleStart);
canvas.addEventListener('mousemove', handleDrawing);
canvas.addEventListener('mouseup', handleEnd);
canvas.addEventListener('mouseout', handleEnd);

canvas.addEventListener('touchstart', handleStart);
canvas.addEventListener('touchmove', handleDrawing);
canvas.addEventListener('touchend', handleEnd);

clearButton.addEventListener('click', handleClear);
detailsButton.addEventListener("click", toggleDetails);

collapsible.style.display = "none";


function handleStart(event) {
    event.preventDefault();
    state.mousedown = true;
    handleDrawing(event);
    collapsible.style.display = "block";
}

function handleDrawing(event) {
    event.preventDefault();

    if (!state.mousedown) return;

    const mousePos = getMousePos(event);

    ctx.lineTo(mousePos.x, mousePos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(mousePos.x, mousePos.y);

    submitCanvas();
}

function handleEnd(event) {
    event.preventDefault();
    state.mousedown = false;
    ctx.beginPath();
}

function handleClear(event) {
    event.preventDefault();
    clearCanvas();
    document.getElementById("prediction").innerHTML = "---"
    collapsible.style.display = "none";
}

function getMousePos(event) {
    const clientX = event.clientX || event.touches[0].clientX;
    const clientY = event.clientY || event.touches[0].clientY;
    const { offsetLeft, offsetTop } = event.target;
    const canvasX = clientX - offsetLeft;
    const canvasY = clientY - offsetTop;

    return { x: canvasX, y: canvasY };
}

function clearCanvas() {

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getImageBase64() {
    return canvas.toDataURL();
}

function submitCanvas() {
    const imgData64 = getImageBase64();
    var url = window.location.protocol + '//' + window.location.hostname + ':' + window.location.port;
    fetch(url + '/image64', {
        method: "POST",
        body: JSON.stringify({
            imageDataBase64: imgData64,
        }),
        headers: {
            "Content-type": "application/json"
        }
    })
        .then((res) => res.json())
        .then((json) => updateElemets(json));
}

function updateElemets(json) {
    document.getElementById("prediction").innerHTML = json["prediction"]
    console.log(json["activations"])

    const newChildren = []
    json["activations"].forEach(element => {
        const ptag = document.createElement("p");
        ptag.innerHTML = element;
        newChildren.push(ptag);
    });
    details.replaceChildren(...newChildren)
}

function toggleDetails() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
        content.style.display = "none";
        document.getElementById("details-button").innerHTML = "Show Details"
    } else {
        content.style.display = "block";
        document.getElementById("details-button").innerHTML = "Hide Details"
    }
}
