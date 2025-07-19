import React from "react";

const Alert = ({ testType, onCancel, onTest }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
      <div className="bg-white p-6 rounded-xl shadow-lg text-center max-w-sm w-full">
        <h2 className="text-xl font-semibold mb-4">
          Do you want to test <span className="text-blue-600">{testType}</span>?
        </h2>
        <div className="flex justify-around mt-4">
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-300 rounded-lg hover:bg-gray-400"
          >
            Cancel
          </button>
          <button
            onClick={onTest}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Test
          </button>
        </div>
      </div>
    </div>
  );
};

export default Alert;
