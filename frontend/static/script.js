document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predict-form").addEventListener("submit", async function (event) {
        event.preventDefault(); // Prevent page reload

        let day = document.getElementById("day").value;
        let dish = document.getElementById("dish").value;

        let response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ day: day, dish: dish }),
        });

        let result = await response.json();
        document.getElementById("result").innerText = `Predicted Sales: ${result.predicted_sales}`;
    });
});
