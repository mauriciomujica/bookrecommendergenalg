const mensajeInput = document.getElementById('mensaje');
const enviarBoton = document.getElementById('enviarMensaje');
const chatOutput = document.getElementById('chat-output');
const datasetButton = document.getElementById('dataset-button');
const chatbotButton = document.getElementById('chatbot-button');
const datasetSection = document.getElementById('dataset-section');
const chatbotSection = document.getElementById('chatbot-section');
// const datasetForm = document.getElementById('dataset-form'); // Commented out - not used
const datasetResults = document.getElementById('dataset-results');
const welcomeForm = document.getElementById('welcome-form');
const welcomeSection = document.getElementById('welcome-section');
const mainButtons = document.getElementById('main-buttons');
const backButton = document.getElementById('back-button');
const loadingContainer = document.getElementById('loading-container');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
let currentUserId = null;
let eventSource = null;
let currentSearchType = null; // 'book' or 'author'
let selectedItem = null; // selected book or author
let optionListenersAttached = false;

// Initialize chatbot if on separate page
if (chatbotSection && chatbotSection.style.display === 'block' && !mainButtons) {
    // Set currentUserId from URL if not set
    if (!currentUserId) {
        const urlParams = new URLSearchParams(window.location.search);
        const userId = urlParams.get('user_id');
        if (userId) {
            currentUserId = userId;
        }
    }
    document.getElementById('chat-output').innerHTML = '<div class="asistente">¡Hola! Soy tu asistente de recomendación de lecturas. ¿En qué puedo ayudarte hoy?</div>';
    resetChatOptions();
    attachOptionListeners();
    document.getElementById('gemini-section').style.display = 'none';
}

// Handle welcome form submission
if (welcomeForm) {
    welcomeForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userId = document.getElementById('user-id').value.trim();
        if (!userId) {
            alert('Por favor, ingrese un número de usuario.');
            return;
        }
        currentUserId = userId;
        // Set chatbot link
        const chatbotLink = document.getElementById('chatbot-link');
        if (chatbotLink) {
            chatbotLink.href = `chatbot.html?user_id=${currentUserId}`;
        }
        welcomeSection.style.display = 'none';
        if (mainButtons) {
            mainButtons.style.display = 'block';
            datasetSection.style.display = 'block';
            chatbotSection.style.display = 'none';
        } else {
            // for separate pages
            if (datasetSection) datasetSection.style.display = 'block';
            if (chatbotSection) chatbotSection.style.display = 'block';
        }

        // Auto-load dataset for the user if dataset section exists
        if (datasetSection) {
            await loadDataset(userId);
        }

        // If showing chatbot section, initialize it
        if (chatbotSection && chatbotSection.style.display === 'block') {
            document.getElementById('chat-output').innerHTML = '<div class="asistente">¡Hola! Soy tu asistente de recomendación de lecturas. ¿En qué puedo ayudarte hoy?</div>';
            resetChatOptions();
            attachOptionListeners();
            document.getElementById('gemini-section').style.display = 'none';
        }
    });
}

function agregarMensaje(texto, clase, esHTML = false) {
    const div = document.createElement('div');
    div.className = clase;
    if (esHTML) {
        div.innerHTML = texto;
    } else {
        div.textContent = texto;
    }
    chatOutput.appendChild(div);
    chatOutput.scrollTop = chatOutput.scrollHeight;
}

if (enviarBoton) {
    enviarBoton.addEventListener('click', async () => {
        // Normal Gemini chat
        const mensaje = mensajeInput.value;
        if (mensaje.trim() === '') {
            alert('Por favor, escribe un mensaje.');
            return;
        }

        // Mostrar el mensaje del usuario en el chat
        agregarMensaje(`Tú: ${mensaje}`, 'usuario');
        mensajeInput.value = ''; // Limpiar el input

        try {
            // Enviar el mensaje al servidor Flask
            const response = await fetch('http://127.0.0.1:5000/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mensaje: mensaje })
            });

            const data = await response.json();

            if (response.ok) {
                // Mostrar la respuesta del chatbot
                agregarMensaje(`Asistente: ${data.respuesta}`, 'asistente', true);
            } else {
                // Manejar errores
                agregarMensaje(`Error: ${data.error}`, 'asistente');
            }

        } catch (error) {
            console.error('Error:', error);
            agregarMensaje('Error: No se pudo conectar con el servidor.', 'asistente');
        }
    });
}

if (datasetButton) {
    datasetButton.addEventListener('click', () => {
        datasetSection.style.display = 'block';
        chatbotSection.style.display = 'none';
    });
}

if (chatbotButton) {
    chatbotButton.addEventListener('click', () => {
        datasetSection.style.display = 'none';
        chatbotSection.style.display = 'block';
        // Reset chat to initial state
        document.getElementById('chat-output').innerHTML = '<div class="asistente">¡Hola! Soy tu asistente de recomendación de lecturas. ¿En qué puedo ayudarte hoy?</div>';
        resetChatOptions();
        attachOptionListeners();
        document.getElementById('gemini-section').style.display = 'none'; // Hide Gemini input by default
    });
}

if (backButton) {
    backButton.addEventListener('click', () => {
        welcomeSection.style.display = 'block';
        mainButtons.style.display = 'none';
        datasetSection.style.display = 'none';
        chatbotSection.style.display = 'none';
        datasetResults.innerHTML = '';
        document.getElementById('dataset-selection').innerHTML = '';
        document.getElementById('user-id').value = '';
        currentUserId = null;
    });
}

// Function to load dataset for a user
async function loadDataset(userId) {
    const selectionContainer = document.getElementById('dataset-selection');
    selectionContainer.innerHTML = ''; // clear previous selection UI

    // Show loading spinner
    loadingContainer.style.display = 'block';
    datasetResults.innerHTML = '';

    try {
        const response = await fetch(`/search-csv?user_id=${encodeURIComponent(userId)}`);
        if (response.ok) {
            const data = await response.json();
            if (data.results && Object.keys(data.results).length > 0) {
                // Create checkbox grid from object values (titles), with keys (ISBNs) as values
                const results = Object.entries(data.results);
                const listHtml = results.map(([isbn, title], idx) => {
                    return `<div class="checkbox-item"><label><input type="checkbox" data-index="${idx}" value="${escapeHtml(isbn)}"> ${escapeHtml(title)}</label></div>`;
                }).join('');

                datasetResults.innerHTML = `<div class="checkbox-list">${listHtml}</div>`;

                // Add control buttons: Select all, Deselect all, Send Selected
                const controlsHtml = document.createElement('div');
                controlsHtml.className = 'checkbox-controls';

                const selectAllBtn = document.createElement('button');
                selectAllBtn.type = 'button';
                selectAllBtn.id = 'select-all';
                selectAllBtn.textContent = 'Seleccionar todo';
                selectAllBtn.addEventListener('click', () => {
                    datasetResults.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
                });

                const deselectAllBtn = document.createElement('button');
                deselectAllBtn.type = 'button';
                deselectAllBtn.id = 'deselect-all';
                deselectAllBtn.textContent = 'Deseleccionar todo';
                deselectAllBtn.addEventListener('click', () => {
                    datasetResults.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
                });

                const sendBtn = document.createElement('button');
                sendBtn.type = 'button';
                sendBtn.id = 'send-selected';
                sendBtn.textContent = 'Obtener recomendaciones';
                sendBtn.addEventListener('click', async () => {
                    const checked = Array.from(datasetResults.querySelectorAll('input[type="checkbox"]:checked'));
                    const chosen = checked.map(cb => cb.value);

                    // Show progress bar
                    progressContainer.style.display = 'block';
                    progressFill.style.width = '0%';
                    progressText.textContent = 'Iniciando algoritmo...';

                    // Start SSE connection for progress updates
                    eventSource = new EventSource('/progress');
                    eventSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        progressFill.style.width = data.progress + '%';
                        progressText.textContent = data.message;
                        if (data.progress >= 100) {
                            eventSource.close();
                        }
                    };

                    try {
                        const resp = await fetch('/selected', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ selected: chosen, userid: userId }) // Include userid in the payload
                        });
                        const data = await resp.json();

                        if (resp.ok && data.results) {
                            displayRecommendations(data.results, selectionContainer);
                        } else {
                            selectionContainer.innerHTML = `<p>Error: ${data.error || 'Unknown error occurred'}</p>`;
                        }
                    } catch (err) {
                        console.error(err);
                        selectionContainer.innerHTML = '<p>Error sending selected items.</p>';
                    } finally {
                        // Hide progress bar
                        progressContainer.style.display = 'none';
                        if (eventSource) {
                            eventSource.close();
                        }
                    }
                });

                controlsHtml.appendChild(selectAllBtn);
                controlsHtml.appendChild(deselectAllBtn);
                controlsHtml.appendChild(sendBtn);

                // Append controls after the list
                datasetResults.appendChild(controlsHtml);

            } else {
                datasetResults.innerHTML = '<p>No results found for the given User ID.</p>';
            }
        } else {
            datasetResults.innerHTML = '<p>Error searching the CSV file.</p>';
        }
    } catch (error) {
        console.error('Error:', error);
        datasetResults.innerHTML = '<p>Unable to connect to the server.</p>';
    } finally {
        // Hide loading spinner
        loadingContainer.style.display = 'none';
    }
}

// Function to display book recommendations in a card with navigation
function displayRecommendations(books, container) {
    if (!books || books.length === 0) {
        container.innerHTML = '<p>No recommendations available.</p>';
        return;
    }

    let currentIndex = 0;

    function updateDisplay() {
        const book = books[currentIndex];
        const title = escapeHtml(book.title || `Libro ${currentIndex + 1}`);
        const synopsis = escapeHtml(book.synopsis || 'Sinopsis no disponible');

        const html = `
            <div class="recommendation-card">
                 <div class="card-header">
                     <h3>Recomendación ${currentIndex + 1} de ${books.length}</h3>
                 </div>
                 <div class="card-content">
                     <h4>${title}</h4>
                     <p>${synopsis}</p>
                 </div>
                 <div class="card-navigation">
                     <button id="prev-btn" ${currentIndex === 0 ? 'disabled' : ''}>Anterior</button>
                     <button id="next-btn" ${currentIndex === books.length - 1 ? 'disabled' : ''}>Siguiente</button>
                 </div>
             </div>
         `;

        container.innerHTML = html;

        // Add event listeners after setting innerHTML
        document.getElementById('prev-btn').addEventListener('click', () => {
            if (currentIndex > 0) {
                currentIndex--;
                updateDisplay();
            }
        });
        document.getElementById('next-btn').addEventListener('click', () => {
            if (currentIndex < books.length - 1) {
                currentIndex++;
                updateDisplay();
            }
        });
    }

    updateDisplay();
}

// Small helper to escape HTML when injecting values
function escapeHtml(unsafe) {
    return String(unsafe)
        .replace(/&/g, '&')
        .replace(/</g, '<')
        .replace(/>/g, '>')
        .replace(/"/g, '"')
        .replace(/'/g, '&#039;');
}

function attachOptionListeners() {
    if (optionListenersAttached) return;
    optionListenersAttached = true;

    // Chat options handlers
    document.getElementById('estado-pedido').addEventListener('click', () => {
        agregarMensaje('Estado del pedido', 'usuario');
        agregarMensaje('Asistente: Esta funcionalidad estará disponible próximamente.', 'asistente');
        resetChatOptions();
    });

    document.getElementById('locales-cercanos').addEventListener('click', () => {
        agregarMensaje('Locales más cercanos', 'usuario');
        agregarMensaje('Asistente: Esta funcionalidad estará disponible próximamente.', 'asistente');
        resetChatOptions();
    });

    document.getElementById('soporte').addEventListener('click', () => {
        agregarMensaje('Soporte', 'usuario');
        agregarMensaje('Asistente: Esta funcionalidad estará disponible próximamente.', 'asistente');
        resetChatOptions();
    });

    document.getElementById('toggle-gemini').addEventListener('click', () => {
        agregarMensaje('Gemini', 'usuario');
        const geminiSection = document.getElementById('gemini-section');
        if (geminiSection.style.display === 'none' || geminiSection.style.display === '') {
            geminiSection.style.display = 'block';
        } else {
            geminiSection.style.display = 'none';
        }
    });

    document.getElementById('inventario').addEventListener('click', () => {
        agregarMensaje('Inventario', 'usuario');
        agregarMensaje('Asistente: Muy bien! ¿En qué estás interesado?', 'asistente');
        // Hide the main chat options and show sub-options
        document.getElementById('chat-options').style.display = 'none';
        document.getElementById('sub-options').innerHTML = `
            <button class="sub-option-btn" id="buscar-libro">Buscar por libro</button>
            <button class="sub-option-btn" id="buscar-autor">Buscar por autor</button>
        `;
        document.getElementById('sub-options').style.display = 'block';

        // Attach listeners to the newly created buttons
        document.getElementById('buscar-libro').addEventListener('click', () => {
            console.log('Buscar libro clicked');
            agregarMensaje('Buscar por libro', 'usuario');
            currentSearchType = 'book';
            agregarMensaje('Asistente: Escribe el nombre del libro:', 'asistente');
            document.getElementById('sub-options').style.display = 'none';
            document.getElementById('search-input-container').style.display = 'block';
            document.getElementById('search-input').placeholder = 'Escribe el nombre del libro...';
            document.getElementById('search-input').focus();
        });

        document.getElementById('buscar-autor').addEventListener('click', () => {
            console.log('Buscar autor clicked');
            agregarMensaje('Buscar por autor', 'usuario');
            currentSearchType = 'author';
            agregarMensaje('Asistente: Escribe el nombre del autor:', 'asistente');
            document.getElementById('sub-options').style.display = 'none';
            document.getElementById('search-input-container').style.display = 'block';
            document.getElementById('search-input').placeholder = 'Escribe el nombre del autor...';
            document.getElementById('search-input').focus();
        });
    });
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Search functionality with autocomplete
const searchInput = document.getElementById('search-input');

if (searchInput) {
    const debouncedSearch = debounce(async () => {
        const query = searchInput.value.trim();
        if (query.length < 2) {
            document.getElementById('search-list').innerHTML = '';
            return;
        }

        try {
            const endpoint = currentSearchType === 'book' ? '/search-books' : '/search-authors';
            const response = await fetch(`${endpoint}?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (response.ok && data.results) {
                // Populate datalist for autocomplete
                const datalist = document.getElementById('search-list');
                datalist.innerHTML = data.results.map(item => `<option value="${escapeHtml(item)}">`).join('');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }, 300); // 300ms delay

    searchInput.addEventListener('input', debouncedSearch);

    searchInput.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
            const query = searchInput.value.trim();
            if (!query) {
                alert('Por favor, escribe algo para buscar.');
                return;
            }

            try {
                const endpoint = currentSearchType === 'book' ? '/search-books' : '/search-authors';
                const response = await fetch(`${endpoint}?q=${encodeURIComponent(query)}`);
                const data = await response.json();

                if (response.ok && data.results && data.results.length > 0) {
                    // Check if query matches one of the results (case insensitive)
                    const matchedItem = data.results.find(item => item.toLowerCase() === query.toLowerCase());
                    if (matchedItem) {
                        selectedItem = matchedItem;
                        // Hide search
                        document.getElementById('search-input-container').style.display = 'none';
                        // Clear input and datalist
                        document.getElementById('search-input').value = '';
                        document.getElementById('search-list').innerHTML = '';
                        // Show actions
                        if (currentSearchType === 'book') {
                            agregarMensaje(selectedItem, 'usuario');
                            agregarMensaje(`Asistente: Has seleccionado el libro "${selectedItem}". ¿Qué deseas hacer?`, 'asistente');
                            document.getElementById('item-options').innerHTML = `
                                <button class="action-btn" id="breve-sinopsis">Breve sinopsis</button>
                                <button class="action-btn" id="informacion">Información</button>
                                <button class="action-btn" id="usar-recomendacion">Usar como recomendación</button>
                                <button class="action-btn" id="inicio">Inicio</button>
                            `;
                            document.getElementById('item-options').style.display = 'block';
                        } else if (currentSearchType === 'author') {
                            agregarMensaje(selectedItem, 'usuario');
                            agregarMensaje(`Asistente: Has seleccionado el autor "${selectedItem}". ¿Qué deseas hacer?`, 'asistente');
                            document.getElementById('item-options').innerHTML = `
                                <button class="action-btn" id="informacion-autor">Información</button>
                                <button class="action-btn" id="libros-disponibles">Libros disponibles</button>
                                <button class="action-btn" id="inicio">Inicio</button>
                            `;
                            document.getElementById('item-options').style.display = 'block';
                        }
                    } else {
                        // No exact match, but show suggestions
                        const suggestionsList = '<ul class="search-results-list">' + data.results.map(item => `<li class="clickable-search-result" data-item="${escapeHtml(item)}">${escapeHtml(item)}</li>`).join('') + '</ul>';
                        agregarMensaje(`Asistente: No encontré una coincidencia exacta para "${query}". Aquí tienes algunas sugerencias:${suggestionsList}`, 'asistente', true);

                        // Attach click listeners to suggestions
                        setTimeout(() => {
                            document.querySelectorAll('.clickable-search-result').forEach(li => {
                                li.addEventListener('click', () => {
                                    const item = li.dataset.item;
                                    selectedItem = item;
                                    agregarMensaje(item, 'usuario');
                                    if (currentSearchType === 'book') {
                                        agregarMensaje(`Asistente: Has seleccionado el libro "${item}". ¿Qué deseas hacer?`, 'asistente');
                                        document.getElementById('item-options').innerHTML = `
                                            <button class="action-btn" id="breve-sinopsis">Breve sinopsis</button>
                                            <button class="action-btn" id="informacion">Información</button>
                                            <button class="action-btn" id="usar-recomendacion">Usar como recomendación</button>
                                            <button class="action-btn" id="inicio">Inicio</button>
                                        `;
                                    } else if (currentSearchType === 'author') {
                                        agregarMensaje(`Asistente: Has seleccionado el autor "${item}". ¿Qué deseas hacer?`, 'asistente');
                                        document.getElementById('item-options').innerHTML = `
                                            <button class="action-btn" id="informacion-autor">Información</button>
                                            <button class="action-btn" id="libros-disponibles">Libros disponibles</button>
                                            <button class="action-btn" id="inicio">Inicio</button>
                                        `;
                                    }
                                    document.getElementById('item-options').style.display = 'block';
                                    document.getElementById('chat-options').style.display = 'none';
                                });
                            });
                        }, 0);

                        // Hide search input after showing suggestions
                        document.getElementById('search-input-container').style.display = 'none';
                        document.getElementById('search-input').value = '';
                        document.getElementById('search-list').innerHTML = '';
                    }
                } else {
                    agregarMensaje('Asistente: No se encontraron resultados para tu búsqueda.', 'asistente');
                }
            } catch (error) {
                console.error('Error:', error);
                agregarMensaje('Asistente: Error al buscar. Inténtalo de nuevo.', 'asistente');
            }
        }
    });
}

// Handle item selection
document.addEventListener('click', async (event) => {
    if (event.target.classList.contains('clickable-recommendation')) {
        selectedItem = event.target.dataset.book;
        currentSearchType = 'recommendation';
        agregarMensaje(selectedItem, 'usuario');
        agregarMensaje(`Asistente: Has seleccionado el libro "${selectedItem}". ¿Qué deseas hacer?`, 'asistente');
        document.getElementById('item-options').innerHTML = `
            <button class="action-btn" id="breve-sinopsis">Breve sinopsis</button>
            <button class="action-btn" id="informacion">Información</button>
            <button class="action-btn" id="inicio">Inicio</button>
        `;
        document.getElementById('item-options').style.display = 'block';
        document.getElementById('chat-options').style.display = 'none';
    } else if (event.target.classList.contains('item-option-btn')) {
        selectedItem = event.target.dataset.item;
        console.log('Item selected:', selectedItem, 'currentSearchType:', currentSearchType);
        document.getElementById('search-input-container').style.display = 'none';
        document.getElementById('item-options').style.display = 'none';

        if (currentSearchType === 'book') {
            agregarMensaje(`Asistente: Has seleccionado el libro "${selectedItem}". ¿Qué deseas hacer?`, 'asistente');
            document.getElementById('item-options').innerHTML = `
                <button class="action-btn" id="breve-sinopsis">Breve sinopsis</button>
                <button class="action-btn" id="informacion">Información</button>
                <button class="action-btn" id="usar-recomendacion">Usar como recomendación</button>
                <button class="action-btn" id="inicio">Inicio</button>
            `;
            document.getElementById('item-options').style.display = 'block';
        } else if (currentSearchType === 'author') {
            agregarMensaje(`Asistente: Has seleccionado el autor "${selectedItem}". ¿Qué deseas hacer?`, 'asistente');
            document.getElementById('item-options').innerHTML = `
                <button class="action-btn" id="informacion-autor">Información</button>
                <button class="action-btn" id="libros-disponibles">Libros disponibles</button>
                <button class="action-btn" id="inicio">Inicio</button>
            `;
            document.getElementById('item-options').style.display = 'block';
            console.log('Author options displayed, selectedItem is:', selectedItem);
        }
    }
});

// Handle final actions
document.addEventListener('click', async (event) => {
    if (event.target.id === 'breve-sinopsis') {
        agregarMensaje('Breve sinopsis', 'usuario');
        try {
            const response = await fetch('/generate-synopsis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ book: selectedItem })
            });
            const data = await response.json();
            if (response.ok) {
                agregarMensaje(`Asistente: ${data.synopsis}`, 'asistente');
            } else {
                agregarMensaje('Asistente: Error al generar la sinopsis.', 'asistente');
            }
        } catch (error) {
            agregarMensaje('Asistente: Error al conectar con el servidor.', 'asistente');
        }
        resetChatOptions();
    } else if (event.target.id === 'informacion') {
        // For books, show sub-options for information source
        document.getElementById('item-options').style.display = 'none';
        document.getElementById('sub-options').innerHTML = `
            <button class="sub-option-btn" id="info-database">Base de datos</button>
            <button class="sub-option-btn" id="info-gemini">Gemini</button>
        `;
        document.getElementById('sub-options').style.display = 'block';
    } else if (event.target.id === 'info-database') {
        agregarMensaje('Base de datos', 'usuario');
        try {
            const response = await fetch(`/get-book-info-csv?book=${encodeURIComponent(selectedItem)}`);
            const data = await response.json();
            if (response.ok) {
                const infoHtml = `
                    <div style="text-align: center; margin: 10px 0;">
                        <img src="${data.image_url}" alt="Portada del libro" style="max-width: 200px; height: auto; border: 1px solid #ddd; margin-bottom: 10px;">
                        <p><strong>Título:</strong> ${escapeHtml(data.title)}</p>
                        <p><strong>Autor:</strong> ${escapeHtml(data.author)}</p>
                        <p><strong>Año:</strong> ${escapeHtml(data.year)}</p>
                        <p><strong>Editorial:</strong> ${escapeHtml(data.publisher)}</p>
                        <p><strong>ISBN:</strong> ${escapeHtml(data.isbn)}</p>
                    </div>
                `;
                agregarMensaje(`Asistente: Información del libro desde la base de datos:${infoHtml}`, 'asistente', true);
            } else {
                agregarMensaje('Asistente: Error al obtener información de la base de datos.', 'asistente');
            }
        } catch (error) {
            agregarMensaje('Asistente: Error al conectar con el servidor.', 'asistente');
        }
        resetChatOptions();
    } else if (event.target.id === 'info-gemini') {
        agregarMensaje('Gemini', 'usuario');
        try {
            const response = await fetch('/generate-info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ book: selectedItem })
            });
            const data = await response.json();
            if (response.ok) {
                agregarMensaje(`Asistente: ${data.info}`, 'asistente');
            } else {
                agregarMensaje('Asistente: Error al obtener información.', 'asistente');
            }
        } catch (error) {
            agregarMensaje('Asistente: Error al conectar con el servidor.', 'asistente');
        }
        resetChatOptions();
    } else if (event.target.id === 'usar-recomendacion') {
        agregarMensaje('Usar como recomendación', 'usuario');
        if (!currentUserId) {
            agregarMensaje('Asistente: Debes iniciar sesión para usar esta funcionalidad.', 'asistente');
            resetChatOptions();
        } else {
            // Hide item options and show rating
            document.getElementById('item-options').style.display = 'none';
            document.getElementById('rating-section').style.display = 'block';
            // Generate radio buttons
            const ratingOptions = document.getElementById('rating-options');
            ratingOptions.innerHTML = '';
            for (let i = 1; i <= 10; i++) {
                const label = document.createElement('label');
                label.style.margin = '0 5px';
                label.innerHTML = `<input type="radio" name="rating" value="${i}" style="margin-right: 5px;"> ${i}`;
                ratingOptions.appendChild(label);
            }
            // Attach submit listener
            document.getElementById('submit-rating').onclick = () => {
                const selected = document.querySelector('input[name="rating"]:checked');
                if (!selected) {
                    alert('Por favor, selecciona una calificación.');
                    return;
                }
                const rating = selected.value;
                // Send to server
                fetch('/add-recommendation', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: currentUserId, book: selectedItem, rating: rating })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        agregarMensaje('Asistente: Gracias por tu calificación. El libro ha sido agregado a tus recomendaciones.', 'asistente');
                        if (data.recommendations && data.recommendations.length > 0) {
                            const recList = '<ul class="books-list">' + data.recommendations.map(book => `<li class="clickable-recommendation" data-book="${escapeHtml(book)}">${escapeHtml(book)}</li>`).join('') + '</ul>';
                            agregarMensaje(`Asistente: Basado en tu calificación, aquí tienes algunas recomendaciones:${recList}`, 'asistente', true);
                        }
                    } else {
                        agregarMensaje('Asistente: Error al agregar recomendación.', 'asistente');
                    }
                    document.getElementById('rating-section').style.display = 'none';
                    resetChatOptions();
                })
                .catch(() => {
                    agregarMensaje('Asistente: Error al conectar con el servidor.', 'asistente');
                    document.getElementById('rating-section').style.display = 'none';
                    resetChatOptions();
                });
            };
        }
    } else if (event.target.id === 'informacion-autor') {
        agregarMensaje('Información', 'usuario');
        try {
            const response = await fetch('/generate-info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ author: selectedItem })
            });
            const data = await response.json();
            if (response.ok) {
                agregarMensaje(`Asistente: ${data.info}`, 'asistente');
            } else {
                agregarMensaje('Asistente: Error al obtener información.', 'asistente');
            }
        } catch (error) {
            agregarMensaje('Asistente: Error al conectar con el servidor.', 'asistente');
        }
        resetChatOptions();
    } else if (event.target.id === 'libros-disponibles') {
        agregarMensaje('Libros disponibles', 'usuario');
        console.log('Libros disponibles clicked, selectedItem:', selectedItem);
        if (!selectedItem) {
            console.error('selectedItem is undefined!');
            agregarMensaje('Asistente: Error: No se ha seleccionado un autor.', 'asistente');
            resetChatOptions();
            return;
        }
        try {
            const response = await fetch(`/get-books-by-author?author=${encodeURIComponent(selectedItem)}`);
            const data = await response.json();
            console.log('Response data:', data);
            if (response.ok && data.books && data.books.length > 0) {
                const booksList = '<ul class="books-list">' + data.books.map(book => `<li class="clickable-book" data-book="${escapeHtml(book)}">${escapeHtml(book)}</li>`).join('') + '</ul>';
                agregarMensaje(`Asistente: Libros disponibles de ${selectedItem} (${data.books.length}):${booksList}`, 'asistente', true);

                // Attach click listeners to clickable books after the message is added
                setTimeout(() => {
                    document.querySelectorAll('.clickable-book').forEach(li => {
                        li.addEventListener('click', () => {
                            const book = li.dataset.book;
                            selectedItem = book;
                            agregarMensaje(book, 'usuario');
                            agregarMensaje(`Asistente: Has seleccionado el libro "${book}". ¿Qué deseas hacer?`, 'asistente');
                            document.getElementById('item-options').innerHTML = `
                                <button class="action-btn" id="breve-sinopsis">Breve sinopsis</button>
                                <button class="action-btn" id="informacion">Información</button>
                                <button class="action-btn" id="usar-recomendacion">Usar como recomendación</button>
                                <button class="action-btn" id="inicio">Inicio</button>
                            `;
                            document.getElementById('item-options').style.display = 'block';
                            // Hide main options if visible
                            document.getElementById('chat-options').style.display = 'none';
                        });
                    });
                }, 0);
            } else {
                agregarMensaje('Asistente: No se encontraron libros para este autor.', 'asistente');
            }
        } catch (error) {
            console.error('Error fetching books:', error);
            agregarMensaje('Asistente: Error al obtener libros.', 'asistente');
        }
        resetChatOptions();
    }
});

function resetChatOptions() {
    // Hide all dynamic elements
    document.getElementById('sub-options').style.display = 'none';
    document.getElementById('search-input-container').style.display = 'none';
    document.getElementById('item-options').style.display = 'none';
    document.getElementById('gemini-section').style.display = 'none';
    document.getElementById('rating-section').style.display = 'none';
    document.getElementById('search-input').value = '';
    document.getElementById('search-list').innerHTML = '';
    currentSearchType = null;
    selectedItem = null;

    // Show main chat options (they're now inline in chat)
    const chatOptions = document.getElementById('chat-options');
    if (chatOptions) {
        chatOptions.style.display = 'block';
    }
}

// Handle final actions - now includes inicio button
document.addEventListener('click', async (event) => {
    if (event.target.id === 'inicio') {
        agregarMensaje('Inicio', 'usuario');
        // Clear chat and show initial options
        document.getElementById('chat-output').innerHTML = '<div class="asistente">¡Hola! Soy tu asistente de recomendación de lecturas. ¿En qué puedo ayudarte hoy?</div>';
        resetChatOptions();
        return;
    }
    // ... existing code for other actions
});