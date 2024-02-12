"use client";

import { useState } from "react";

export default function Home() {
  const [review, setReview] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeSentiment = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review }),
      });
      const data = await response.json();
      setSentiment(data.sentiment);
    } catch (error) {
      setError("An error occurred. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <h1 className="text-3xl font-bold mb-4">Sentiment Analysis</h1>
      <div className="w-full max-w-lg">
        <label htmlFor="review" className="block text-gray-700 mb-2">
          Enter your product review:
        </label>
        <textarea
          id="review"
          className="block w-full p-2 border border-gray-300 rounded mb-4"
          rows="4"
          value={review}
          onChange={(e) => setReview(e.target.value)}
        ></textarea>
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded"
          onClick={analyzeSentiment}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        {error && <p className="text-red-500 mt-2">{error}</p>}
        {sentiment && (
          <div className="mt-4">
            <h2 className="text-xl font-bold mb-2">Sentiment:</h2>
            <p>{sentiment}</p>
          </div>
        )}
      </div>
    </div>
  );
}
