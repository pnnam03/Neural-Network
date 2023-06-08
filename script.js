const message = document.querySelector("span");

const ACTIVATION_RATE = 0.07;
let numbers = [];
let topology = [];
let weight = [];
let outputVals = [];

readTextFile("net.dat");

/*------- initialize the net -------*/
var n = numbers[0];
for (let i = 0; i < n; ++i) {
  topology.push(numbers[i + 1]);
}

let cnt = n + 1;
for (let i = 0; i < n - 1; ++i) {
  let tmpi = [];
  for (let j = 0; j < topology[i] + 1; ++j) {
    let tmpj = [];
    for (let k = 0; k < topology[i + 1]; ++k) {
      tmpj.push(numbers[cnt]);
      cnt++;
    }
    tmpi.push(tmpj);
  }
  weight.push(tmpi);
}

/*------- mouse event -------- */
const scaleCanvas = document.querySelector("#scaling-canvas");
const scaleCtx = scaleCanvas.getContext("2d");
const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");

//
let isDrawing, isErasing;
const drawColor = "white";
const eraseColor = "black";
const lineWidth = 20;
const clearBtn = document.querySelector(".clear-canvas");

canvas.addEventListener("mouseout", (e) => {
  if (e.which === 1) {
    // left mouse button up
    isDrawing = false;
  }
  if (e.which === 3) {
    // right mouse button up
    isErasing = false;
  }
});

canvas.addEventListener("mouseup", (e) => {
  if (e.which === 1) {
    // left mouse button up
    isDrawing = false;
  }
  if (e.which === 3) {
    // right mouse button up
    isErasing = false;
  }
  imgData = getImageData();

  message.innerText = "Your number is " + getResult(imgData);
});

canvas.addEventListener("mousemove", (e) => {
  if (isDrawing) {
    ctx.fillStyle = drawColor;
    ctx.fillRect(
      e.offsetX - lineWidth / 2,
      e.offsetY - lineWidth / 2,
      lineWidth,
      lineWidth
    );
    ctx.fill();
  } else if (isErasing) {
    ctx.fillStyle = eraseColor;
    ctx.fillRect(
      e.offsetX - lineWidth / 2,
      e.offsetY - lineWidth / 2,
      lineWidth,
      lineWidth
    );
    ctx.fill();
  }
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mousedown", (e) => {
  if (e.which === 1) {
    // left mouse button up
    isDrawing = true;
  }
  if (e.which === 3) {
    // right mouse button up
    isErasing = true;
  }
  ctx.beginPath();
});

clearBtn.addEventListener("click", () => {
  ctx.fillStyle = eraseColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  scaleCtx.fillStyle = eraseColor;
  scaleCtx.fillRect(0, 0, scaleCanvas.width, scaleCanvas.height);

  message.innerText = "Your number is: ";
});

window.addEventListener("load", () => {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;

  ctx.fillStyle = eraseColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  scaleCanvas.width = scaleCanvas.height = 28;

  scaleCtx.fillStyle = eraseColor;
  scaleCtx.fillRect(0, 0, canvas.width, canvas.height);

  numbers = [];
  readTextFile("test.inp");
  showImage(numbers);
});
function activationFunction(x) {
  return Math.max(x, 0.0) * ACTIVATION_RATE;
}

function getResult(inputVals) {
  let tmp = [];
  outputVals = [];
  for (let i = 0; i < topology[0]; ++i) {
    tmp.push(inputVals[i] / 255);
  }
  tmp.push(1); // bias neuron
  outputVals.push(tmp);

  for (let i = 1; i < n; ++i) {
    let tmpi = [];
    for (let j = 0; j < topology[i]; ++j) {
      let sumWeightedInput = 0.0;

      for (let k = 0; k < topology[i - 1] + 1; ++k) {
        let w = weight[i - 1][k][j];
        sumWeightedInput += outputVals[i - 1][k] * w;
      }

      tmpi.push(activationFunction(sumWeightedInput));
    }
    tmpi.push(1);
    outputVals.push(tmpi);
  }

  let maxVal = outputVals[n - 1][0];
  let result = 0;
  for (let i = 0; i < topology[n - 1]; ++i) {
    if (outputVals[n - 1][i] > maxVal) {
      maxVal = outputVals[n - 1][i];
      result = i;
    }
  }
  return result;
}

function readTextFile(file) {
  var rawFile = new XMLHttpRequest();
  rawFile.open("GET", file, false);
  rawFile.onreadystatechange = function () {
    if (rawFile.readyState === 4) {
      if (rawFile.status === 200 || rawFile.status == 0) {
        var allText = rawFile.responseText;
        numbers = allText.split(/\s+/).map(Number);
      }
    }
  };
  rawFile.send(null);
}

function getImageData() {
  let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  imgData = scaleImageData(imgData, 28 / canvas.width);

  let pixels = imgData.data;
  var result1 = [];
  var result = [];
  for (let i = 0; i < pixels.length; i += 4) {
    var avg = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
    result1.push(avg);
    result.push(0);
  }

  var dx = 27,
    dy = 27;
  for (let x = 0; x < 28; ++x)
    for (let y = 0; y < 28; ++y)
      if (result1[x * 28 + y] > 0) {
        dx = Math.min(dx, x);
        dy = Math.min(dy, y);
      }

  for (let x = dx; x < 28; ++x)
    for (let y = dy; y < 28; ++y) {
      var newPos = (x - dx) * 28 + (y - dy);
      var oldPos = x * 28 + y;
      result[newPos] = result1[oldPos];
    }
  return result;
}

function scaleImageData(imageData, scale) {
  // Canvas for scaling
  scaleCtx.drawImage(canvas, 0, 0, 28, 28);
  var scaledImageData = scaleCtx.getImageData(
    0,
    0,
    scaleCanvas.width,
    scaleCanvas.height
  );
  // scaleCtx.scale(1 / scale, 1 / scale);
  return scaledImageData;
}
