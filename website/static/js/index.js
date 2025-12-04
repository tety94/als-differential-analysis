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
      const response = await fetch("/als-differential-analysis/predict", {
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
            let cc = 'SLA';
            if (values.class == 0) {
                cc = 'NON SLA';
            }

            html += `
                <li class="list-group-item">
                    <strong>${model}</strong>: ${cc}, probabilità = ${(values.probability * 100).toFixed(2)}%
            `;

            // --- Aggiungi SHAP explanation se presente ---
            if (values.shap_explanation) {
                const shap = values.shap_explanation;

                html += `<ul class="list-group mt-2">
                            <li class="list-group-item"><strong>Base value:</strong> ${shap.base_value.toFixed(4)}</li>
                            <li class="list-group-item"><strong>Somma SHAP:</strong> ${shap.sum_shap.toFixed(4)}</li>
                         </ul>`;

                html += `<ul class="list-group mt-2">`;
                for (const [feature, shap_val] of Object.entries(shap.shap_values)) {
                    html += `<li class="list-group-item p-1">
                                 <strong>${feature}</strong>: ${Number(shap_val).toFixed(4)}
                             </li>`;
                }
                html += `</ul>`;
            }


            html += `</li>`;
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


//per il change del modello
  const modelSelect = document.getElementById('model_type');

    function updateFieldsVisibility() {
        const selectedModel = modelSelect.value;

        // Nascondi tutti i campi prima
        document.querySelectorAll('.nice-form-group[class*="third_level"], .nice-form-group[class*="neurologi"], .nice-form-group[class*="medici_base"]').forEach(field => {
            field.style.display = 'none';
        });

        // Mostra solo i campi del modello selezionato
        document.querySelectorAll('.nice-form-group.' + selectedModel).forEach(field => {
            field.style.display = 'block';
        });
    }

    // Esegui al caricamento
    updateFieldsVisibility();

    // Esegui al cambio del modello
    modelSelect.addEventListener('change', updateFieldsVisibility);
});
