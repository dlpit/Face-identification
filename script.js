const container = document.querySelector('#container');
const fileInput = document.querySelector('#file-input');

async function loadTrainingData() {
  const labels = ['Bong', 'Linh', 'Phong', 'NgocKem'];

  const faceDescriptors = [];
  for (const label of labels) {
    const descriptors = [];
    for (let i = 1; i <= 4; i++) {
      const image = await faceapi.fetchImage(`/data/${label}/${i}.jpg`);
      const detections = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
      descriptors.push(detections.descriptor);
    }
    faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
    Toastify({
      text: `Tải dữ liệu ${label} thành công!`,
      backgroundColor: 'linear-gradient(to right, #00b09b, #96c93d)',
    }).showToast();
  }
  return faceDescriptors;
}

let faceMatcher;
async function init() {  
  await Promise.all([
    await faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models')
  ]);

  const trainingData = await loadTrainingData();
  faceMatcher = new faceapi.FaceMatcher(trainingData, 0.6);

  Toastify({
    text: 'Tải model thành công!',
    backgroundColor: 'linear-gradient(to right, #00b09b, #96c93d)',
  }).showToast();
}

init();

fileInput.addEventListener('change', async (e) => {
  const file = fileInput.files[0];

  const image = await faceapi.bufferToImage(file);
  const canvas = faceapi.createCanvasFromMedia(image);

  container.innerHTML = '';
  container.append(image);
  container.append(canvas);

  const size = { width: image.width, height: image.height };
  faceapi.matchDimensions(canvas, size);

  const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
  const resizedDetections = faceapi.resizeResults(detections, size);

  for (const detection of resizedDetections) {
    const box = detection.detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, { 
      label: faceMatcher.findBestMatch(detection.descriptor).toString(),
    });
    drawBox.draw(canvas);
  }
});