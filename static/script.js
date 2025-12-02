async function analyzeText() {
    const text = document.getElementById("inputText").value.trim();
    const resultDiv = document.getElementById("result");
    const loading = document.getElementById("loading");

    if (text.length < 5) {
        resultDiv.style.display = "block";
        resultDiv.className = "result-error";
        resultDiv.innerText = "Please enter at least 5 characters.";
        return;
    }

    resultDiv.style.display = "none";
    loading.style.display = "block";

    try {
        const formData = new FormData();
        formData.append("text", text);

        // Replace this with your deployed backend URL if you have one
        const backendURL = "http://127.0.0.1:8000/predict";

        const response = await fetch(backendURL, {
            method: "POST",
            body: formData
        });

        const data = await response.json().catch(() => ({
            prediction: "error",
            funny_response: "Invalid response from server"
        }));

        loading.style.display = "none";
        resultDiv.style.display = "block";

        if (data.prediction === "human") {
            resultDiv.className = "result-human";
            resultDiv.innerHTML = `✅ Text detected as: HUMAN<br>${data.funny_response}`;
        } else if (data.prediction === "ai") {
            resultDiv.className = "result-ai";
            resultDiv.innerHTML = `🤖 Text detected as: AI<br>${data.funny_response}`;
        } else {
            resultDiv.className = "result-error";
            resultDiv.innerText = "Unexpected response from server.";
        }

    } catch (err) {
        loading.style.display = "none";
        resultDiv.style.display = "block";
        resultDiv.className = "result-error";
        resultDiv.innerText = "Server error. Please check backend is running.";
        console.error(err);
    }
}
