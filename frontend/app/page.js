"use client";

import React, { useState } from "react";

function App() {
  const [startYear, setStartYear] = useState(2021);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState([]);

  const handleSliderChange = (e) => {
    setStartYear(parseInt(e.target.value));
  };

  const handleSeedChange = (e) => {
    setSeed(parseInt(e.target.value));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPredictions([]);
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/predict?start_year=${startYear}&seed=${seed}`
      );
      if (!response.ok) throw new Error("Prediction failed");
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Return main HTML
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold text-center mb-6">
        NHL 2025-26 Season Predictor
      </h1>
      <p className="text-center mb-4">
        Select the start year for training data and click to run the ML model.
      </p>
      <div className="mx-auto max-w-sm my-6">
        <label
          htmlFor="startYear"
          className="block text-sm font-medium text-gray-700"
        >
          Start Season Year ({startYear} to 2025)
        </label>
        <input
          type="range"
          id="startYear"
          min="2021"
          max="2023"
          step="1"
          value={startYear}
          onChange={handleSliderChange}
          className="w-full mt-2"
        />
        <p className="text-center mt-2">
          Using data from {startYear}-{startYear + 1} onwards
        </p>
      </div>
      <div className="mx-auto max-w-sm my-6">
        <label
          htmlFor="seed"
          className="block text-sm font-medium text-gray-700"
        >
          Simulation Seed (1-100) for Variance: {seed}
        </label>
        <input
          type="range"
          id="seed"
          min="1"
          max="100"
          step="1"
          value={seed}
          onChange={handleSeedChange}
          className="w-full mt-2"
        />
      </div>
      <button
        onClick={handlePredict}
        className="block mx-auto bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded cursor-pointer"
      >
        Run Prediction
      </button>
      {loading && (
        <div className="text-center mt-4">
          Loading... (Request may take up to a minute due to Render free plan
          spinning down with inactivity)
        </div>
      )}
      {error && (
        <div className="text-center text-red-500 mt-4">Error: {error}</div>
      )}
      {predictions.length > 0 && (
        <table className="w-full mt-6 bg-white shadow-md rounded-lg overflow-hidden">
          <thead className="bg-gray-200">
            <tr>
              <th className="px-4 py-2">Rank</th>
              <th className="px-4 py-2">Team</th>
              <th className="px-4 py-2">Expected Position</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((row, index) => (
              <tr key={index}>
                <td className="border px-4 py-2">{row.predicted_rank}</td>
                <td className="border px-4 py-2">{row.team}</td>
                <td className="border px-4 py-2">
                  {row.expected_position.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;
