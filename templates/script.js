document.addEventListener("DOMContentLoaded", function () {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const fileNameText = document.getElementById("file-name");
    const uploadBtn = document.getElementById("upload-btn");

    // Drag and Drop Events
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.style.background = "#e6f7ff";
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.style.background = "#f9f9f9";
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.style.background = "#f9f9f9";

        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    // File Input Change Event
    fileInput.addEventListener("change", () => {
        const file = fileInput.files[0];
        handleFile(file);
    });

    // Handle File Upload
    function handleFile(file) {
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = () => {
                previewImage.src = reader.result;
                previewContainer.style.display = "block";
                fileNameText.textContent = file.name;
                uploadBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        } else {
            alert("Please upload a valid image file.");
        }
    }

    // Upload Button Click Event
    uploadBtn.addEventListener("click", () => {
        const file = fileInput.files[0];

        if (!file) {
            alert("No file selected!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            document.body.innerHTML = data;
        })
        .catch(error => {
            console.error("Error uploading file:", error);
            alert("Error uploading file.");
        });
    });
});
