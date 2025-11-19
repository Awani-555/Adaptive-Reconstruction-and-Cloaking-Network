const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");

startBtn.onclick = async () => {
  const res = await fetch("/start", { method: "POST" });
  const data = await res.json();
  statusEl.textContent = `Status: ${data.status}`;
};

stopBtn.onclick = async () => {
  const res = await fetch("/stop", { method: "POST" });
  const data = await res.json();
  statusEl.textContent = `Status: ${data.status}`;
};
