/**
 * Основные функции JavaScript для системы дубляжа видео
 */

document.addEventListener('DOMContentLoaded', function() {
    // Обработка выхода со страницы во время выполнения задачи
    window.addEventListener('beforeunload', function(e) {
        // Если на странице есть прогресс-бар, который не скрыт, показываем предупреждение
        const progressBar = document.querySelector('.progress-bar-animated:not(.d-none)');
        if (progressBar) {
            // Chrome требует установить returnValue
            e.returnValue = 'Задача все еще выполняется. Вы уверены, что хотите покинуть страницу?';
            // Для других браузеров
            return 'Задача все еще выполняется. Вы уверены, что хотите покинуть страницу?';
        }
    });

    // Предварительная проверка файла перед отправкой
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            validateFile(this);
        });
    }

    // Инициализация всплывающих подсказок
    initializeTooltips();
});

/**
 * Проверка загружаемого файла
 * @param {HTMLInputElement} fileInput - Поле ввода файла
 */
function validateFile(fileInput) {
    if (!fileInput.files.length) return;
    
    const file = fileInput.files[0];
    const maxSize = 500 * 1024 * 1024; // 500 МБ
    const uploadButton = document.getElementById('upload-button');
    const fileError = document.getElementById('file-error');
    
    // Создаем элемент для ошибки, если его нет
    if (!fileError && fileInput.parentNode) {
        const errorDiv = document.createElement('div');
        errorDiv.id = 'file-error';
        errorDiv.className = 'invalid-feedback d-block';
        fileInput.parentNode.appendChild(errorDiv);
    }
    
    // Проверка размера файла
    if (file.size > maxSize) {
        if (fileError) fileError.textContent = 'Файл слишком большой. Максимальный размер: 500 МБ';
        fileInput.value = '';
        if (uploadButton) uploadButton.disabled = true;
        return false;
    }
    
    // Проверка типа файла
    const fileType = file.name.split('.').pop().toLowerCase();
    const allowedTypes = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
    
    if (!allowedTypes.includes(fileType)) {
        if (fileError) fileError.textContent = 'Недопустимый формат файла. Разрешены: MP4, AVI, MOV, MKV, WEBM';
        fileInput.value = '';
        if (uploadButton) uploadButton.disabled = true;
        return false;
    }
    
    // Очищаем сообщение об ошибке, если файл прошел проверку
    if (fileError) fileError.textContent = '';
    if (uploadButton) uploadButton.disabled = false;
    
    return true;
}

/**
 * Инициализация всплывающих подсказок Bootstrap
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Форматирование размера файла для отображения
 * @param {number} bytes - Размер в байтах
 * @returns {string} Отформатированный размер с единицей измерения
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Байт';
    
    const k = 1024;
    const sizes = ['Байт', 'КБ', 'МБ', 'ГБ'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Копирование текста в буфер обмена
 * @param {string} text - Текст для копирования
 * @param {function} callback - Функция обратного вызова после копирования
 */
function copyToClipboard(text, callback) {
    navigator.clipboard.writeText(text).then(function() {
        if (callback) callback(true);
    }, function(err) {
        console.error('Не удалось скопировать текст: ', err);
        if (callback) callback(false);
    });
} 