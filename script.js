const { TFLiteModel, get_model } = require('./src/backbone');
const { mediapipe_detection, draw, extract_coordinates, load_json_file } = require('./src/landmarks_extraction');
const { SEQ_LEN, THRESH_HOLD } = require('./src/config');
const cv = require('opencv4nodejs');

const s2p_map = Object.fromEntries(Object.entries(load_json_file("src/sign_to_prediction_index_map.json")).map(([k, v]) => [k.toLowerCase(), v]));
const p2s_map = Object.fromEntries(Object.entries(load_json_file("src/sign_to_prediction_index_map.json")).map(([k, v]) => [v, k]));
const encoder = x => s2p_map[x.toLowerCase()];
const decoder = x => p2s_map[x];

const models_path = [
  './models/islr-fp16-192-8-seed_all42-foldall-last.h5',
];
const models = models_path.map(get_model);

// Load weights from the weights file.
for (let i = 0; i < models.length; i++) {
  models[i].load_weights(models_path[i]);
}

async function real_time_asl() {
  const tflite_keras_model = new TFLiteModel(models);
  const sequence_data = [];
  const cap = new cv.VideoCapture(0);

  const mp_holistic = new mp.Holistic({
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  while (cap.read()) {
    let frame = cap.read();
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB);
    
    let image;
    let results;
    [image, results] = mediapipe_detection(frame, mp_holistic);
    draw(image, results);

    let landmarks;
    try {
      landmarks = extract_coordinates(results);
    } catch (error) {
      landmarks = new cv.Mat.zeros(468 + 21 + 33 + 21, 3, cv.CV_32F);
    }
    sequence_data.push(landmarks);

    let sign = "";

    // Generate the prediction for the given sequence data.
    if (sequence_data.length % SEQ_LEN === 0) {
      const prediction = await tflite_keras_model.predict(sequence_data);

      if (Math.max(...prediction.numpy(), axis=-1) > THRESH_HOLD) {
        sign = prediction.argmax(axis=-1).squeeze();
      }

      sequence_data.length = 0;
    }

    image = cv.flip(image, 1);
    cv.putText(image, `${sequence_data.length}`, new cv.Point(3, 35),
      cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Vec3(0, 0, 255), 2, cv.LINE_AA);
    image = cv.flip(image, 1);

    if (sign !== "" && decoder(sign) && !res.includes(decoder(sign))) {
      res.unshift(decoder(sign));
    }

    const height = image.rows;
    const width = image.cols;

    const white_column = new cv.Mat.ones(height / 8, width, cv.CV_8UC3).mul(255);
    image = cv.vconcat(white_column, image);

    cv.putText(image, `${res.join(', ')}`, new cv.Point(3, 65),
      cv.FONT_HERSHEY_SIMPLEX, 3, new cv.Vec3(0, 0, 0), 2, cv.LINE_AA);

    cv.imshow('Webcam Feed', image);

    // Wait for a key to be pressed.
    if (cv.waitKey(10) === 113) { // ASCII value of 'q'
      break;
    }
  }

  cap.release();
  cv.destroyAllWindows();
}

real_time_asl();
