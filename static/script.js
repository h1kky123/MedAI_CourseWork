document.addEventListener('DOMContentLoaded', () => {
    const messagesDiv = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    function parseMarkdown(text) {
        return text
            // Горизонтальная линия
            .replace(/^---$/gm, '<hr class="md-hr">')
            // Заголовки **Текст:** секция-заголовок
            .replace(/\*\*([^*]+):\*\*/g, '<span class="md-label">$1</span>')
            // Жирный **текст**
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Курсив _текст_
            .replace(/_([^_]+)_/g, '<em>$1</em>')
            // Курсив *текст* (одиночная звезда)
            .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
            // Переносы строк
            .replace(/\n/g, '<br>');
    }

    //Рендер секций препарата
    function renderDrugInfo(answer) {
        // Разбиваем по секциям **Название:** текст
        const sectionRegex = /\*\*([^*]+):\*\*\s*([\s\S]*?)(?=\n\*\*|\n---|\n={3}|$)/g;
        const sections = [];
        let match;
        let lastIndex = 0;

        // Заголовок
        const firstSectionStart = answer.search(/\*\*[^*]+:\*\*/);
        if (firstSectionStart > 0) {
            const title = answer.slice(0, firstSectionStart).trim();
            if (title) sections.push({ type: 'title', text: title });
        }

        while ((match = sectionRegex.exec(answer)) !== null) {
            sections.push({ type: 'section', label: match[1], content: match[2].trim() });
            lastIndex = sectionRegex.lastIndex;
        }

        // Остаток после последней секции
        const remainder = answer.slice(lastIndex).trim();
        if (remainder) sections.push({ type: 'remainder', text: remainder });

        if (sections.length <= 1) return null;

        const sectionColors = {
            'Показания': 'section-green',
            'Активные вещества': 'section-blue',
            'Противопоказания': 'section-red',
            'Применение': 'section-purple',
            'Побочные эффекты': 'section-orange',
        };

        let html = '';
        sections.forEach(s => {
            if (s.type === 'title') {
                html += `<div class="drug-title">${parseMarkdown(s.text)}</div>`;
            } else if (s.type === 'section') {

                const colorClass = sectionColors[s.label] || '';

                if (s.label.trim().toLowerCase() === 'источник') {
                    html += `<div class="drug-source">
                                <a href="${s.content.trim()}" target="_blank" class="drug-source-link">
                                    Открыть страницу препарата
                                </a>
                             </div>`;
                    return;
                }

                html += `
                <div class="drug-section ${colorClass}">
                    <div class="drug-section-header">
                        <span class="drug-section-label">${s.label}</span>
                    </div>
                    <div class="drug-section-body">${parseMarkdown(s.content)}</div>
                </div>`;
            } else if (s.type === 'remainder') {
                const lines = s.text.split('<br>').join('\n').split('\n');
                let remainderHtml = '';
                lines.forEach(line => {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('•')) {
                        remainderHtml += `<div class="other-form-item">${trimmed.slice(1).trim()}</div>`;
                    } else if (trimmed) {
                        remainderHtml += `<div>${parseMarkdown(trimmed)}</div>`;
                    }
                });
                html += `<div class="drug-remainder">${remainderHtml}</div>`;
            }
        });
        return html;
    }

    //Рендер списка аналогов
    function renderAnalogs(answer) {
        if (!answer.includes('аналог') && !answer.includes('Аналог')) return null;

        let html = '';
        const lines = answer.split('\n');
        let inList = false;
        let currentHeader = '';

        lines.forEach(line => {
            // Заголовок препарата
            if (line.startsWith('**Аналоги для препарата')) {
                html += `<div class="drug-title">${parseMarkdown(line)}</div>`;
            }
            // Подзаголовок группы
            else if (line.startsWith('**') && line.endsWith('**')) {
                if (inList) { html += '</div>'; inList = false; }
                html += `<div class="analog-group-header">${parseMarkdown(line)}</div>`;
                html += `<div class="analog-list">`;
                inList = true;
            }
            // Элемент списка
            else if (/^\d+\./.test(line.trim())) {
                html += `<div class="analog-item">${parseMarkdown(line.trim())}</div>`;
            }
            // Обычный текст
            else if (line.trim()) {
                if (inList) { html += '</div>'; inList = false; }
                html += `<div class="analog-text">${parseMarkdown(line)}</div>`;
            }
        });
        if (inList) html += '</div>';
        return html || null;
    }

    //Рендер выбора препарата
    function renderSelection(answer, sources) {
        const lines = answer.split('\n');
        const intro = lines[0];
        let html = `<div class="selection-intro">${intro}</div><div class="selection-list">`;

        sources.forEach(source => {
            const displayName = source.trade_name;
            html += `<button class="selection-btn" onclick="window.sendMessage('Инструкция для ${displayName}')">
                        <span class="selection-pill"></span>
                        <span>${displayName}</span>
                     </button>`;
        });

        html += `</div><div class="selection-hint">Нажмите на препарат или уточните запрос</div>`;
        return html;
    }

    //Рендер симптом/общий
    function renderSymptom(answer) {
        const lines = answer.split('\n');
        let html = '';
        lines.forEach(line => {
            if (/^\*\*\d+\./.test(line) || /^\d+\./.test(line.trim())) {
                html += `<div class="symptom-item">${parseMarkdown(line.trim())}</div>`;
            } else if (line.trim()) {
                html += `<div>${parseMarkdown(line)}</div>`;
            } else {
                html += '<br>';
            }
        });
        return html;
    }

    //Главная функция рендера бота
    window.addBotMessage = function(data) {
        const div = document.createElement('div');
        div.className = 'message bot';

        const intentLabels = {
            'symptom':  { label: 'Поиск по симптому',        icon: '🔍' },
            'drug_info':{ label: 'Информация о препарате',    icon: '📋' },
            'analogs':  { label: 'Поиск аналогов',            icon: '🔄' },
            'general':  { label: 'Общий запрос',              icon: '💬' },
            'error':    { label: 'Ошибка',                    icon: '❌' },
            'selection':{ label: 'Уточните препарат',         icon: '🔎' },
        };

        const intentInfo = intentLabels[data.type] || { label: 'Ответ ИИ', icon: '🤖' };
        let html = `<div class="intent-badge">
                        <span>${intentInfo.icon}</span>
                        <span>${intentInfo.label}</span>
                    </div>`;

        // Выбираем рендерер по типу
        let bodyHtml = null;

        if (data.type === 'selection') {
            bodyHtml = renderSelection(data.answer, data.sources || []);
        } else if (data.type === 'drug_info') {
            bodyHtml = renderDrugInfo(data.answer);
        } else if (data.type === 'analogs') {
            bodyHtml = renderAnalogs(data.answer);
        } else if (data.type === 'symptom') {
            bodyHtml = renderSymptom(data.answer);
        }

        // Fallback
        if (!bodyHtml) {
            bodyHtml = `<div class="bot-text">${parseMarkdown(data.answer)}</div>`;
        }

        html += `<div class="bot-body">${bodyHtml}</div>`;

        // Источники только для symptom/general
        if (data.sources && data.sources.length > 0 && ['symptom', 'general'].includes(data.type)) {
            html += `<div class="sources-container">
                        <div class="sources-title">Найдено в базе:</div>
                        <div>`;
            data.sources.forEach(source => {
                if (source.source_url) {
                    html += `<a href="${source.source_url}" target="_blank" class="source-tag">💊 ${source.trade_name}</a>`;
                } else {
                    html += `<span class="source-tag">💊 ${source.trade_name}</span>`;
                }
            });
            html += `</div></div>`;
        }

        div.innerHTML = html;
        messagesDiv.appendChild(div);
        scrollToBottom();
    };

    //Отправка сообщения
    window.sendMessage = async function(overrideText) {
        const text = overrideText || userInput.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        if (!overrideText) {
            userInput.value = '';
            userInput.style.height = 'auto';
        }

        sendBtn.disabled = true;
        const loadingId = addLoadingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text, top_k: 5 })
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            document.getElementById(loadingId)?.remove();
            addBotMessage(data);
        } catch (error) {
            document.getElementById(loadingId)?.remove();
            addMessage('Ошибка соединения с сервером.', 'bot');
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    };

    window.addMessage = function(text, sender) {
        const div = document.createElement('div');
        div.className = `message ${sender}`;
        div.textContent = text;
        messagesDiv.appendChild(div);
        scrollToBottom();
    };

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const div = document.createElement('div');
        div.id = id;
        div.className = 'message bot';
        div.innerHTML = `<div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>`;
        messagesDiv.appendChild(div);
        scrollToBottom();
        return id;
    }

    function scrollToBottom() {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    });

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            window.sendMessage();
        }
    });
});