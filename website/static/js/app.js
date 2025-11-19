document.addEventListener("DOMContentLoaded", function() {
    const currentLang = document.documentElement.dataset.lang;
    console.log('currentLang')
    console.log(currentLang)

    const itItem = document.querySelector(".lang-it");
    const enItem = document.querySelector(".lang-en");

    if (currentLang === "it") {
        // nasconde la lingua attuale e mostra l'altra
        itItem.style.display = "none";
        enItem.style.display = "block";
    } else if (currentLang === "en") {
        enItem.style.display = "none";
        itItem.style.display = "block";
    }
});