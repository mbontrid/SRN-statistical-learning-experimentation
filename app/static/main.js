window.onload = function () {

    const input = document.querySelector('.import-box__import-input');
    const button = document.querySelector('.import-box__import-button');
    const form = document.querySelector('.parameters-form');

    button.addEventListener('click', () => input.click());

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            button.textContent = input.files[0].name;
        }
    });

    form.addEventListener('submit', () => {
        document.getElementById('loading-overlay').style.display = 'flex';
    });

    form.addEventListener("submit", async function (e) {
        e.preventDefault();

        const formData = new FormData(form);
        const response = await fetch("/train_stream", { method: "POST", body: formData });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";  // ← accumulate across chunks

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true }); // ← stream: true is important

            // Split only on complete SSE messages
            const messages = buffer.split("\n\n");
            buffer = messages.pop(); // ← keep the last incomplete part

            for (const message of messages) {
                const line = message.trim();
                if (!line.startsWith("data:")) continue;

                try {
                    const data = JSON.parse(line.slice(5).trim()); // "data:".length === 5

                    if (data.plot) {
                        console.log("plot traces:", data.plot.data);  // ← check x/y arrays here
                        Plotly.newPlot("plot-container", data.plot.data, data.plot.layout);
                    }

                    if (data.done) {
                        document.getElementById("epoch-text").innerText = "Training complete";
                        document.getElementById("loading-overlay").style.display = "none";
                        return;
                    }

                    document.getElementById("epoch-text").innerText = "Epoch: " + data.epoch;
                    document.getElementById("progress-fill").style.width = data.progress + "%";

                } catch (err) {
                    console.error("Failed to parse SSE message:", line, err);
                }
            }
        }
    });
};