// Sets up custom dropdown behavior for all select wrappers
document.addEventListener("DOMContentLoaded", () => {
    const selectWrappers = document.querySelectorAll('.select-wrapper');

    selectWrappers.forEach(wrapper => {
        const selectBox = wrapper.querySelector('.custom-select');
        if (!selectBox) return;

        const selectedText = selectBox.querySelector('.selected');
        const options = selectBox.querySelector('.options');
        const optionList = selectBox.querySelectorAll('.option');

        if (!optionList.length) return;

        const defaultOption = optionList[0];
        selectedText.textContent = defaultOption.textContent;
        defaultOption.classList.add('selected');

        // Toggle options display on select box click
        selectBox.addEventListener('click', () => {
            options.style.display = options.style.display === 'block' ? 'none' : 'block';
            selectBox.classList.toggle('open');
        });

        // Update selected option and hide options on option click
        optionList.forEach(option => {
            option.addEventListener('click', () => {
                selectedText.textContent = option.textContent;
                optionList.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');

                // Show/hide custom prompt textarea
                if (selectBox.id === 'style') {
                    const customWrapper = document.getElementById('custom-prompt-wrapper');
                    const promptNameInput = document.getElementById('custom_prompt_name');
                    const promptTextarea = document.getElementById('custom_prompt');
                    const deleteBtn = document.getElementById('delete-prompt-btn');
                    
                    if (option.textContent.includes('Custom')) {
                        customWrapper.style.display = 'block';
                        
                        // Check if it's a saved prompt
                        if (option.hasAttribute('data-is-saved-prompt')) {
                            const promptName = option.textContent.replace('Custom: ', '');
                            try {
                                const saved = JSON.parse(localStorage.getItem('saved_custom_prompts') || '{}');
                                if (saved[promptName]) {
                                    promptTextarea.value = saved[promptName];
                                    if (promptNameInput) promptNameInput.value = promptName;
                                    if (deleteBtn) deleteBtn.style.display = 'inline-block';
                                }
                            } catch (e) {}
                        } else {
                            // Regular 'Custom...' option
                            if (promptNameInput && option.textContent === 'Custom...') {
                                promptNameInput.value = '';
                                if (deleteBtn) deleteBtn.style.display = 'none';
                            }
                        }
                    } else {
                        customWrapper.style.display = 'none';
                    }
                }

                // Show/hide settings based on translator and OCR selections
                if (selectBox.id === 'translator' || selectBox.id === 'ocr') {
                    const translatorText = document.querySelector('#translator .selected').textContent;
                    const ocrText = document.querySelector('#ocr .selected').textContent;
                    
                    const copilotSettings = document.getElementById('copilot-settings');
                    const geminiSettings = document.getElementById('gemini-settings');
                    const freellmSettings = document.getElementById('freellm-settings');

                    copilotSettings.style.display = (translatorText === 'Local LLM') ? 'block' : 'none';
                    geminiSettings.style.display = translatorText.startsWith('Gemini') ? 'block' : 'none';
                    freellmSettings.style.display = (translatorText === 'FreeLLM' || ocrText === 'FreeLLM-Vision') ? 'block' : 'none';
                }

                if (selectBox.id === 'pipeline_mode') {
                    const mode = option.textContent;
                    const ocrWrapper = document.getElementById('ocr_wrapper');
                    const fontWrapper = document.getElementById('font_wrapper');
                    const workersSettings = document.getElementById('gemini-workers-settings');
                    const workersNote = document.getElementById('gemini-workers-note');
                    
                    if (mode.includes('Gemini Full')) {
                        if (ocrWrapper) ocrWrapper.style.display = 'none';
                        if (fontWrapper) fontWrapper.style.display = 'none';
                        if (workersSettings) workersSettings.style.display = 'block';
                        if (workersNote) workersNote.style.display = 'block';
                    } else if (mode.includes('Gemini Hybrid')) {
                        if (ocrWrapper) ocrWrapper.style.display = 'none';
                        if (fontWrapper) fontWrapper.style.display = 'block';
                        if (workersSettings) workersSettings.style.display = 'block';
                        if (workersNote) workersNote.style.display = 'none';
                    } else {
                        if (ocrWrapper) ocrWrapper.style.display = 'block';
                        if (fontWrapper) fontWrapper.style.display = 'block';
                        if (workersSettings) workersSettings.style.display = 'none';
                        if (workersNote) workersNote.style.display = 'none';
                    }
                }
            });
        });

        // Hide options when clicking outside the select box
        window.addEventListener('click', e => {
            if (!wrapper.contains(e.target)) {
                options.style.display = 'none';
                selectBox.classList.remove('open');
            }
        });

        // Save dropdown selection to localStorage
        optionList.forEach(option => {
            option.addEventListener('click', () => {
                if (selectBox.id) {
                    localStorage.setItem('select_' + selectBox.id, option.textContent);
                }
            });
        });

        // Restore saved selection on load
        if (selectBox.id) {
            const savedValue = localStorage.getItem('select_' + selectBox.id);
            if (savedValue) {
                optionList.forEach(opt => {
                    if (opt.textContent === savedValue) {
                        selectedText.textContent = savedValue;
                        optionList.forEach(o => o.classList.remove('selected'));
                        opt.classList.add('selected');

                        // Trigger visibility updates for special selects
                        if (selectBox.id === 'style' && savedValue.includes('Custom')) {
                            document.getElementById('custom-prompt-wrapper').style.display = 'block';
                        }
                        if (selectBox.id === 'translator' || selectBox.id === 'ocr') {
                            const translatorText = localStorage.getItem('select_translator') || 'Gemini Flash';
                            const ocrText = localStorage.getItem('select_ocr') || 'Chrome-Lens';
                            
                            const copilotSettings = document.getElementById('copilot-settings');
                            const geminiSettings = document.getElementById('gemini-settings');
                            const freellmSettings = document.getElementById('freellm-settings');
                            
                            copilotSettings.style.display = (translatorText === 'Local LLM') ? 'block' : 'none';
                            geminiSettings.style.display = translatorText.startsWith('Gemini') ? 'block' : 'none';
                            freellmSettings.style.display = (translatorText === 'FreeLLM' || ocrText === 'FreeLLM-Vision') ? 'block' : 'none';
                        }
                    }
                });
            }
        }
    });

    // Trigger initial state for pipeline_mode
    const pipelineModeText = document.querySelector('#pipeline_mode .selected');
    if (pipelineModeText) {
        const mode = pipelineModeText.textContent;
        const ocrWrapper = document.getElementById('ocr_wrapper');
        const fontWrapper = document.getElementById('font_wrapper');
        const workersSettings = document.getElementById('gemini-workers-settings');
        const workersNote = document.getElementById('gemini-workers-note');
        
        if (mode.includes('Gemini Full')) {
            if (ocrWrapper) ocrWrapper.style.display = 'none';
            if (fontWrapper) fontWrapper.style.display = 'none';
            if (workersSettings) workersSettings.style.display = 'block';
            if (workersNote) workersNote.style.display = 'block';
        } else if (mode.includes('Gemini Hybrid')) {
            if (ocrWrapper) ocrWrapper.style.display = 'none';
            if (fontWrapper) fontWrapper.style.display = 'block';
            if (workersSettings) workersSettings.style.display = 'block';
            if (workersNote) workersNote.style.display = 'none';
        } else {
            if (ocrWrapper) ocrWrapper.style.display = 'block';
            if (fontWrapper) fontWrapper.style.display = 'block';
            if (workersSettings) workersSettings.style.display = 'none';
            if (workersNote) workersNote.style.display = 'none';
        }
    }

    // Load saved Gemini Workers from localStorage
    const geminiWorkersInput = document.getElementById('gemini_workers');
    if (geminiWorkersInput) {
        const savedWorkers = localStorage.getItem('gemini_workers');
        if (savedWorkers) {
            geminiWorkersInput.value = savedWorkers;
        }
        geminiWorkersInput.addEventListener('input', () => {
            localStorage.setItem('gemini_workers', geminiWorkersInput.value);
        });
    }

    // Load saved Gemini API key from localStorage
    const geminiKeyInput = document.getElementById('gemini_api_key');
    if (geminiKeyInput) {
        const savedKey = localStorage.getItem('gemini_api_key');
        if (savedKey) {
            geminiKeyInput.value = savedKey;
        }
        geminiKeyInput.addEventListener('input', () => {
            localStorage.setItem('gemini_api_key', geminiKeyInput.value);
        });
    }

    // Load saved FreeLLM API key and URL from localStorage
    const freellmKeyInput = document.getElementById('freellm_api_key');
    if (freellmKeyInput) {
        const savedKey = localStorage.getItem('freellm_api_key');
        if (savedKey) {
            freellmKeyInput.value = savedKey;
        }
        freellmKeyInput.addEventListener('input', () => {
            localStorage.setItem('freellm_api_key', freellmKeyInput.value);
        });
    }

    const freellmBaseUrlInput = document.getElementById('freellm_base_url');
    if (freellmBaseUrlInput) {
        const savedUrl = localStorage.getItem('freellm_base_url');
        if (savedUrl) {
            freellmBaseUrlInput.value = savedUrl;
        }
        freellmBaseUrlInput.addEventListener('input', () => {
            localStorage.setItem('freellm_base_url', freellmBaseUrlInput.value);
        });
    }

    // Load saved Local LLM server URL from localStorage
    const copilotServerInput = document.getElementById('copilot_server');
    if (copilotServerInput) {
        const savedServer = localStorage.getItem('copilot_server');
        if (savedServer) {
            copilotServerInput.value = savedServer;
        }
        copilotServerInput.addEventListener('input', () => {
            localStorage.setItem('copilot_server', copilotServerInput.value);
        });
    }

    // Load saved Local LLM model from localStorage
    const copilotModelInput = document.getElementById('copilot_model_input');
    if (copilotModelInput) {
        const savedModel = localStorage.getItem('copilot_model');
        if (savedModel) {
            copilotModelInput.value = savedModel;
        }
        copilotModelInput.addEventListener('input', () => {
            localStorage.setItem('copilot_model', copilotModelInput.value);
        });
    }

    // Load saved custom prompt from localStorage
    const customPromptInput = document.getElementById('custom_prompt');
    if (customPromptInput) {
        const savedPrompt = localStorage.getItem('custom_prompt');
        if (savedPrompt) {
            customPromptInput.value = savedPrompt;
        }
        customPromptInput.addEventListener('input', () => {
            localStorage.setItem('custom_prompt', customPromptInput.value);
        });
    }

    // Load saved checkbox states from localStorage
    const contextMemoryCheckbox = document.getElementById('context_memory');
    if (contextMemoryCheckbox) {
        const saved = localStorage.getItem('context_memory');
        if (saved !== null) {
            contextMemoryCheckbox.checked = saved === 'true';
        }
        contextMemoryCheckbox.addEventListener('change', () => {
            localStorage.setItem('context_memory', contextMemoryCheckbox.checked);
        });
    }

    const blackBubblesCheckbox = document.getElementById('detect_black_bubbles');
    if (blackBubblesCheckbox) {
        const saved = localStorage.getItem('detect_black_bubbles');
        if (saved !== null) {
            blackBubblesCheckbox.checked = saved === 'true';
        }
        blackBubblesCheckbox.addEventListener('change', () => {
            localStorage.setItem('detect_black_bubbles', blackBubblesCheckbox.checked);
        });
    }
});


// Handles multiple file upload change event
const fileUpload = document.getElementById('file-upload');
if (fileUpload) {
    fileUpload.addEventListener('change', function () {
        const files = this.files;
        const fileList = document.getElementById('file-list');
        const fileText = document.getElementById('file-text');

        if (files.length === 0) {
            fileText.textContent = '📁 Chọn ảnh (có thể chọn nhiều)';
            fileList.innerHTML = '';
            return;
        }

        if (files.length === 1) {
            fileText.textContent = truncateFileName(files[0].name, 25);
            fileList.innerHTML = '';
        } else {
            fileText.textContent = `📁 ${files.length} ảnh đã chọn`;

            // Show file list preview
            fileList.innerHTML = '';
            for (let i = 0; i < Math.min(files.length, 5); i++) {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.textContent = truncateFileName(files[i].name, 30);
                fileList.appendChild(fileItem);
            }

            if (files.length > 5) {
                const moreItem = document.createElement('div');
                moreItem.className = 'file-item more';
                moreItem.textContent = `... và ${files.length - 5} ảnh khác`;
                fileList.appendChild(moreItem);
            }
        }
    });
}

// Truncates file name if it exceeds the maximum length
function truncateFileName(fileName, maxLength) {
    return fileName.length <= maxLength ? fileName : fileName.substr(0, maxLength - 3) + '...';
}

// Updates hidden input fields with selected options
function updateHiddenInputs() {
    const getSelectedText = (id) => {
        const el = document.querySelector(`#${id} .selected`);
        return el ? el.innerText : '';
    };

    document.getElementById("selected_pipeline_mode").value = getSelectedText("pipeline_mode");
    document.getElementById("selected_source_lang").value = getSelectedText("source_lang");
    document.getElementById("selected_language").value = getSelectedText("language");
    document.getElementById("selected_translator").value = getSelectedText("translator");
    document.getElementById("selected_style").value = getSelectedText("style");
    document.getElementById("selected_font").value = getSelectedText("font");
    document.getElementById("selected_ocr").value = getSelectedText("ocr");

    // Gemini API key is optional (backend falls back to service account JSON)
    
    // Validate FreeLLM API key if FreeLLM is selected
    if (translator === 'FreeLLM') {
        const apiKey = document.getElementById('freellm_api_key').value;
        if (!apiKey || apiKey.trim() === '') {
            alert('Vui lòng nhập FreeLLM API Key!');
            return false;
        }
    }

    // Check if files are selected
    const files = document.getElementById('file-upload').files;
    if (files.length === 0) {
        alert('Vui lòng chọn ít nhất 1 ảnh!');
        return false;
    }

    document.querySelector('form').style.display = 'none';
    document.getElementById('loading-img').style.display = 'block';
    document.getElementById('loading-p').style.display = 'block';

    return true;
}

// ============================================
// ASYNC TRANSLATION (Gemini Hybrid/Full mode)
// ============================================

// Override form submission to use fetch for async processing
const translateForm = document.querySelector('form[action="/translate"]');
if (translateForm) {
    translateForm.addEventListener('submit', function(e) {
        // Let updateHiddenInputs handle validation
        // If it returns false, form won't submit anyway
        if (!updateHiddenInputs()) {
            e.preventDefault();
            return;
        }
        
        e.preventDefault();
        
        // Show loading UI
        translateForm.style.display = 'none';
        document.getElementById('loading-img').style.display = 'block';
        document.getElementById('loading-p').style.display = 'block';
        document.getElementById('progress-container').style.display = 'block';
        
        // Submit via fetch
        const formData = new FormData(translateForm);
        
        fetch('/translate', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                // Async mode: server returned JSON, processing in background
                return response.json().then(data => {
                    if (data.status === 'started') {
                        console.log('Async translation started, session:', data.session_id);
                        // Tell server to start processing via Socket.IO
                        if (window._socket) {
                            window._socket.emit('start_translation', { session_id: data.session_id });
                        } else {
                            socket.emit('start_translation', { session_id: data.session_id });
                        }
                        // Show async results container
                        document.getElementById('loading-img').style.display = 'none';
                        document.getElementById('loading-p').style.display = 'none';
                        const resultsContainer = document.getElementById('async-results');
                        if (resultsContainer) {
                            resultsContainer.style.display = 'block';
                            document.getElementById('results-title').textContent = 
                                `🔄 Đang dịch... (0/${data.total} trang)`;
                        }
                    }
                });
            } else {
                // Sync mode (Classic pipeline): server returned HTML
                return response.text().then(html => {
                    document.open();
                    document.write(html);
                    document.close();
                });
            }
        })
        .catch(error => {
            console.error('Translation error:', error);
            alert('Lỗi: ' + error.message);
            translateForm.style.display = 'block';
            document.getElementById('loading-img').style.display = 'none';
            document.getElementById('loading-p').style.display = 'none';
        });
    });
}

// Socket.IO event handlers for async results
const socket = io();
window._socket = socket;  // Make available globally

socket.on('page_result', function(data) {
    console.log('Page result:', data);
    const gallery = document.getElementById('results-gallery');
    if (!gallery) return;
    
    // Update title
    const title = document.getElementById('results-title');
    if (title && data.total > 1) {
        title.textContent = `🔄 Đang dịch... (${data.page_num}/${data.total} trang)`;
    }
    
    // Check if card already exists (retry case) - update instead of append
    const existingCard = gallery.querySelector(`[data-page-name="${data.name}"]`);
    if (existingCard) {
        const img = existingCard.querySelector('.gallery-image');
        if (img) {
            img.src = data.url + '?t=' + Date.now();  // Cache bust
        }
        const retryBtn = existingCard.querySelector('.retry-page-btn');
        if (retryBtn) {
            retryBtn.textContent = '✅';
            retryBtn.style.pointerEvents = 'auto';
            setTimeout(() => { retryBtn.textContent = '🔄'; }, 2000);
        }
        existingCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        return;
    }
    
    // Add image card
    const card = document.createElement('div');
    card.className = 'image-card';
    card.setAttribute('data-page-name', data.name);
    card.innerHTML = `
        <img class="gallery-image" src="${data.url}" alt="${data.name}" loading="lazy">
        <div class="image-info">
            <span class="image-name">${data.name}</span>
            <a href="#" class="retry-page-btn" data-name="${data.name}" data-session="${window._activeSessionId || ''}"
               onclick="retryPage(this, event)" title="Retry trang này">🔄</a>
            <a href="#" class="download-btn" data-url="${data.url}" data-name="${data.name}" 
               onclick="downloadSingle(this, event)">💾</a>
        </div>
    `;
    gallery.appendChild(card);
    
    // Scroll to the new image
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

socket.on('translation_complete', function(data) {
    console.log('Translation complete:', data);
    const title = document.getElementById('results-title');
    if (title) {
        title.textContent = `✨ Kết quả dịch (${data.total} trang)`;
    }
    
    // Show download buttons
    const buttons = document.getElementById('results-buttons');
    if (buttons) {
        buttons.style.display = 'flex';
    }
    
    // Hide progress
    document.getElementById('progress-container').style.display = 'none';
});

socket.on('progress', function(data) {
    const container = document.getElementById('progress-container');
    if (container) container.style.display = 'block';
    
    const bar = document.getElementById('progress-bar');
    const text = document.getElementById('progress-text');
    const phase = document.getElementById('progress-phase');
    
    if (bar) bar.style.width = data.percent + '%';
    if (text) text.textContent = data.message;
    if (phase) phase.textContent = data.phase;
});

// Download single image helper
function downloadSingle(btn, e) {
    e.preventDefault();
    const url = btn.getAttribute('data-url');
    const name = btn.getAttribute('data-name');
    fetch(url)
        .then(r => r.blob())
        .then(blob => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = name + '_translated.jpg';
            a.click();
            URL.revokeObjectURL(a.href);
        });
}

// Retry a single page
function retryPage(btn, e) {
    e.preventDefault();
    const name = btn.getAttribute('data-name');
    const sessionId = btn.getAttribute('data-session') || window._activeSessionId;
    
    if (!sessionId) {
        alert('Không tìm thấy session. Vui lòng dịch lại từ đầu.');
        return;
    }
    
    btn.textContent = '⏳';
    btn.style.pointerEvents = 'none';
    
    const encodedName = encodeURIComponent(name);
    fetch(`/translate/retry-page/${sessionId}/${encodedName}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'retrying_page') {
                btn.textContent = '⏳ Đang xử lý...';
                // page_result socket event will update the image automatically
            } else {
                btn.textContent = '❌';
                setTimeout(() => { btn.textContent = '🔄'; btn.style.pointerEvents = 'auto'; }, 2000);
            }
        })
        .catch(err => {
            console.error('Retry failed:', err);
            btn.textContent = '❌';
            setTimeout(() => { btn.textContent = '🔄'; btn.style.pointerEvents = 'auto'; }, 3000);
        });
}

// Download all as ZIP (async mode)
const zipBtnAsync = document.getElementById('download-zip-async');
if (zipBtnAsync) {
    zipBtnAsync.addEventListener('click', async (e) => {
        e.preventDefault();
        const btn = e.target;
        const originalText = btn.textContent;
        btn.textContent = '⏳ Đang tạo ZIP...';
        btn.style.pointerEvents = 'none';
        
        try {
            const zip = new JSZip();
            const folder = zip.folder('manga_translated');
            const images = document.querySelectorAll('#results-gallery .gallery-image');
            
            for (const img of images) {
                const url = img.getAttribute('src');
                const name = img.getAttribute('alt');
                const response = await fetch(url);
                const blob = await response.blob();
                folder.file(`${name}_translated.jpg`, blob);
            }
            
            const content = await zip.generateAsync({
                type: 'blob', compression: 'DEFLATE',
                compressionOptions: { level: 6 }
            });
            
            const a = document.createElement('a');
            a.href = URL.createObjectURL(content);
            a.download = 'manga_translated.zip';
            a.click();
            URL.revokeObjectURL(a.href);
            
            btn.textContent = '✅ Đã tải xong!';
            setTimeout(() => { btn.textContent = originalText; btn.style.pointerEvents = 'auto'; }, 2000);
        } catch (error) {
            btn.textContent = '❌ Lỗi tạo ZIP';
            setTimeout(() => { btn.textContent = originalText; btn.style.pointerEvents = 'auto'; }, 2000);
        }
    });
}

// ============================================
// SESSION PERSISTENCE & RETRY
// ============================================

// Save session ID to localStorage when translation starts
const origPageResult = socket._callbacks && socket._callbacks['$page_result'];
socket.on('page_result', function(data) {
    // Already handled by existing handler, just save session info
    if (data.page_num && data.total) {
        localStorage.setItem('active_session', JSON.stringify({
            session_id: window._activeSessionId,
            total: data.total,
            lastUpdate: Date.now()
        }));
    }
});

socket.on('translation_complete', function() {
    localStorage.removeItem('active_session');
});

// Store session_id when async translation starts
const origFetchHandler = window._socket;
(function() {
    const origEmit = socket.emit.bind(socket);
    const patchedEmit = function(event, data) {
        if (event === 'start_translation' && data && data.session_id) {
            window._activeSessionId = data.session_id;
            localStorage.setItem('active_session', JSON.stringify({
                session_id: data.session_id,
                total: 0,
                lastUpdate: Date.now()
            }));
        }
        return origEmit(event, data);
    };
    socket.emit = patchedEmit;
})();

// On page load: check for active session and offer resume
window.addEventListener('load', function() {
    const saved = localStorage.getItem('active_session');
    if (!saved) return;
    
    try {
        const session = JSON.parse(saved);
        const age = Date.now() - session.lastUpdate;
        
        // Only resume if session is less than 24 hours old
        if (age > 24 * 60 * 60 * 1000) {
            localStorage.removeItem('active_session');
            return;
        }
        
        // Fetch status from server
        fetch(`/translate/status/${session.session_id}`)
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    localStorage.removeItem('active_session');
                    return;
                }
                
                if (data.completed_count > 0) {
                    // Show resume banner
                    showResumeBanner(data);
                }
            })
            .catch(() => localStorage.removeItem('active_session'));
    } catch(e) {
        localStorage.removeItem('active_session');
    }
});

function showResumeBanner(data) {
    // Hide form, show results
    const form = document.querySelector('form[action="/translate"]');
    if (form) form.style.display = 'none';
    
    const resultsContainer = document.getElementById('async-results');
    if (!resultsContainer) return;
    resultsContainer.style.display = 'block';
    
    const title = document.getElementById('results-title');
    const gallery = document.getElementById('results-gallery');
    const buttons = document.getElementById('results-buttons');
    
    // Show completed images
    gallery.innerHTML = '';
    data.completed.forEach(img => {
        const card = document.createElement('div');
        card.className = 'image-card';
        card.setAttribute('data-page-name', img.name);
        card.innerHTML = `
            <img class="gallery-image" src="${img.url}" alt="${img.name}" loading="lazy">
            <div class="image-info">
                <span class="image-name">${img.name}</span>
                <a href="#" class="retry-page-btn" data-name="${img.name}" data-session="${data.session_id}"
                   onclick="retryPage(this, event)" title="Retry trang này">🔄</a>
                <a href="#" class="download-btn" data-url="${img.url}" data-name="${img.name}"
                   onclick="downloadSingle(this, event)">💾</a>
            </div>
        `;
        gallery.appendChild(card);
    });
    
    if (data.status === 'completed') {
        title.textContent = `✨ Kết quả dịch (${data.completed_count} trang)`;
        buttons.style.display = 'flex';
        localStorage.removeItem('active_session');
    } else if (data.status === 'running') {
        title.textContent = `🔄 Đang dịch... (${data.completed_count}/${data.total} trang)`;
        // Add retry button
        addRetryButton(data, buttons);
    } else {
        // Failed or unknown
        const remaining = data.total - data.completed_count;
        title.textContent = `⚠️ Đã dịch ${data.completed_count}/${data.total} trang (${remaining} trang chưa xong)`;
        buttons.style.display = 'flex';
        addRetryButton(data, buttons);
    }
}

function addRetryButton(data, buttonsContainer) {
    buttonsContainer.style.display = 'flex';
    
    // Check if retry button already exists
    if (document.getElementById('retry-btn')) return;
    
    const retryBtn = document.createElement('a');
    retryBtn.href = '#';
    retryBtn.id = 'retry-btn';
    retryBtn.className = 'green';
    retryBtn.textContent = `🔄 Retry (${data.total - data.completed_count} trang còn lại)`;
    retryBtn.style.background = '#e67e22';
    retryBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        retryBtn.textContent = '⏳ Đang khởi động lại...';
        retryBtn.style.pointerEvents = 'none';
        
        try {
            const resp = await fetch(`/translate/retry/${data.session_id}`, { method: 'POST' });
            const result = await resp.json();
            
            if (result.status === 'retrying') {
                retryBtn.textContent = `🔄 Đang xử lý ${result.remaining} trang...`;
                const title = document.getElementById('results-title');
                if (title) title.textContent = `🔄 Đang dịch lại... (${data.completed_count}/${data.total})`;
                
                // Update active session
                window._activeSessionId = data.session_id;
            } else if (result.status === 'all_completed') {
                retryBtn.textContent = '✅ Tất cả đã hoàn thành!';
                localStorage.removeItem('active_session');
            }
        } catch(err) {
            retryBtn.textContent = '❌ Lỗi: ' + err.message;
            setTimeout(() => {
                retryBtn.textContent = `🔄 Retry`;
                retryBtn.style.pointerEvents = 'auto';
            }, 3000);
        }
    });
    
    // Insert before the back button
    buttonsContainer.insertBefore(retryBtn, buttonsContainer.querySelector('.red'));
}
