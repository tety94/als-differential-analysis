document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predictForm");
  const resultsBox = document.getElementById("resultsBox");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    // raccoglie tutti i dati del form
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    resultsBox.innerHTML = `<div class="alert alert-info">Calcolo in corso...</div>`;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
      });

      const result = await response.json();

      if (result.success) {
        let html = `<h4>Risultati predizione</h4><ul class="list-group">`;
        for (const [model, values] of Object.entries(result.results)) {
          html += `
            <li class="list-group-item">
              <strong>${model}</strong>: classe = ${values.class},
              probabilità = ${(values.probability * 100).toFixed(2)}%
            </li>`;
        }
        html += `</ul>`;
        resultsBox.innerHTML = html;
      } else {
        resultsBox.innerHTML = `<div class="alert alert-danger">Errore nella predizione</div>`;
      }
    } catch (err) {
      console.error(err);
      resultsBox.innerHTML = `<div class="alert alert-danger">Errore di rete o server</div>`;
    }
  });
});
