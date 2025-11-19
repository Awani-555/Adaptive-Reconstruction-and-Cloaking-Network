/*
script.js
Frontend logic for controlling ARCNet invisibility process.

Responsibilities:
- Handle Start/Stop button clicks.
- Send API requests to backend endpoints.
- Update UI status dynamically.
*/

// Get references to DOM elements
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");

// ---------------------------
// START BUTTON HANDLER
// ---------------------------
startBtn.onclick = async () => {
  try {
    // Send POST request to backend '/start' endpoint
    const res = await fetch("/start", { method: "POST" });
    const data = await res.json();

    // Update status text
    statusEl.textContent = `Status: ${data.status}`;
  } catch (err) {
    // Show error in console or UI if request fails
    console.error("Error starting ARCNet:", err);
    statusEl.textContent = "Status: Failed to start.";
  }
};

// ---------------------------
// STOP BUTTON HANDLER
// ---------------------------
stopBtn.onclick = async () => {
  try {
    // Send POST request to backend '/stop' endpoint
    const res = await fetch("/stop", { method: "POST" });
    const data = await res.json();

    // Update status text
    statusEl.textContent = `Status: ${data.status}`;
  } catch (err) {
    console.error("Error stopping ARCNet:", err);
    statusEl.textContent = "Status: Failed to stop.";
  }
};
