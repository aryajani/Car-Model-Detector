import { useState } from "react";
import "./imageUpload.css";

function ImageUpload({ setPrediction }) {

  const [image, setImage] = useState(null);

  const handleChange = (event) => {
    setImage(event.target.files[0]);
  };

  const handleUpload = async () => {

    const formData = new FormData();
    formData.append("file", image);

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    console.log(data);

    setPrediction(data);
  };

  return (
    <div className="upload-container">

      <input type="file" onChange={handleChange} />

      {image && (
        <>
          <img
            src={URL.createObjectURL(image)}
            alt="preview"
            className="preview"
          />

          <button onClick={handleUpload}>
            Classify Car
          </button>
        </>
      )}

    </div>
  );
}

export default ImageUpload;