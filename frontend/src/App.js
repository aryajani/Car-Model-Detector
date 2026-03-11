import { useState } from "react";
import "./App.css";
import ImageUpload from "./imageUpload";
import PredictionResult from "./predictionResult";
import Header from "./header";

function App() {
  const [prediction, setPrediction] = useState("");

  return (
    <div className="app">
      <Header text={"Car Classifier"}/>

      <ImageUpload setPrediction={setPrediction} />

      <PredictionResult prediction={prediction} />

    </div>
  );
}

export default App;