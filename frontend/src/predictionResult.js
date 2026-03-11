import { useState } from "react";
import "./predictionResult.css";

function PredictionResult({ prediction }) {

  const [carInfo, setCarInfo] = useState("");
  const [loading, setLoading] = useState(false);

  if (!prediction) {
    return null;
  }

  const getCarInfo = async () => {

    setLoading(true);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/car-info?name=${prediction.class_name}`
      );

      const data = await response.json();

      setCarInfo(data.info);

    } catch (error) {
      console.error("Error fetching car info:", error);
    }

    setLoading(false);
  };

  return (
    <div className="result-container">

      <h2>Prediction</h2>

      <p className="car-name">{prediction.class_name}</p>

      <button onClick={getCarInfo}>
        Tell me more about this car
      </button>

      {loading && <p>Loading...</p>}

      {carInfo && (
        <div className="car-info">
          <p>{carInfo}</p>
        </div>
      )}

    </div>
  );
}

export default PredictionResult;