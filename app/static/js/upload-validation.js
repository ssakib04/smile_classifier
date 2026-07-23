/** Shared client-side upload validation (10 MB max per file). */
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message, type = "error") {
    let container = document.getElementById("toast-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "toast-container";
        container.className = "toast-container-theme";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast-theme${type === "error" ? " toast-error" : ""}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transition = "opacity 0.3s";
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}

function validateFileInput(input, options = {}) {
    const { clearOnReject = true } = options;
    if (!input?.files?.length) return true;

    const rejected = [];
    for (const file of input.files) {
        if (file.size > MAX_UPLOAD_BYTES) {
            rejected.push(file);
        }
    }

    if (rejected.length === 0) return true;

    const names = rejected.map((f) => `"${f.name}" (${formatFileSize(f.size)})`).join(", ");
    showToast(
        `File exceeds 10 MB limit: ${names}. Please choose smaller images.`,
        "error"
    );

    if (clearOnReject) {
        input.value = "";
    }
    return false;
}

function attachFileSizeValidation(input, options = {}) {
    if (!input) return;
    input.addEventListener("change", () => validateFileInput(input, options));
}
