document.addEventListener('DOMContentLoaded', () => {
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

    // Handle welcome form submission
    welcomeForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userId = document.getElementById('user-id').value.trim();
        if (!userId) {
            alert('Por favor, ingrese un número de usuario.');
            return;
        }
        currentUserId = userId;
        welcomeSection.style.display = 'none';
        mainButtons.style.display = 'block';
        datasetSection.style.display = 'block';
        chatbotSection.style.display = 'none';

        // Auto-load dataset for the user
        await loadDataset(userId);
    });

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

    datasetButton.addEventListener('click', () => {
        datasetSection.style.display = 'block';
        chatbotSection.style.display = 'none';
    });

    chatbotButton.addEventListener('click', () => {
        datasetSection.style.display = 'none';
        chatbotSection.style.display = 'block';
        // Reset chat to initial state
        document.getElementById('chat-output').innerHTML = '<div class="asistente">¡Hola! Soy tu asistente de recomendación de lecturas. ¿En qué puedo ayudarte hoy?</div>';
        resetChatOptions();
        attachOptionListeners();
        document.getElementById('gemini-section').style.display = 'none'; // Hide Gemini input by default
    });

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

    // Commented out - datasetForm is not used in current HTML
    // datasetForm.addEventListener('submit', async (event) => {
    //     event.preventDefault(); // Prevent form submission
    //
    //     const userId = document.getElementById('csv-search').value.trim();
    //     if (!userId) {
    //         datasetResults.innerHTML = '<p>Please enter a User ID.</p>';
    //         return;
    //     }
    //
    //     await loadDataset(userId);
    // });

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
                <style>
                    .recommendation-card {
                        max-width: 500px;
                        margin: 20px auto;
                        padding: 20px;
                        background-color: #fff;
                        border-radius: 10px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        text-align: center;
                    }
                    .card-header h3 {
                        margin: 0 0 15px 0;
                        color: #201f56;
                    }
                    .card-content h4 {
                        margin: 0 0 10px 0;
                        color: #333;
                    }
                    .card-content p {
                        line-height: 1.5;
                        color: #555;
                    }
                    .card-navigation {
                        margin-top: 20px;
                    }
                    .card-navigation button {
                        background-color: #2d6cdf;
                        color: #fff;
                        border: none;
                        border-radius: 6px;
                        padding: 10px 15px;
                        margin: 0 5px;
                        cursor: pointer;
                        transition: background-color 0.3s ease;
                    }
                    .card-navigation button:hover:not(:disabled) {
                        background-color: #1a4e96;
                    }
                    .card-navigation button:disabled {
                        background-color: #ccc;
                        cursor: not-allowed;
                    }
                </style>
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
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function attachOptionListeners() {
        if (optionListenersAttached) return;
        optionListenersAttached = true;

        // Chat options handlers
        document.getElementById('estado-pedido').addEventListener('click', () => {
            agregarMensaje('Asistente: Estado del pedido - Esta funcionalidad estará disponible próximamente.', 'asistente');
            resetChatOptions();
        });

        document.getElementById('locales-cercanos').addEventListener('click', () => {
            agregarMensaje('Asistente: Locales más cercanos - Esta funcionalidad estará disponible próximamente.', 'asistente');
            resetChatOptions();
        });

        document.getElementById('soporte').addEventListener('click', () => {
            agregarMensaje('Asistente: Soporte - Esta funcionalidad estará disponible próximamente.', 'asistente');
            resetChatOptions();
        });

        document.getElementById('toggle-gemini').addEventListener('click', () => {
            const geminiSection = document.getElementById('gemini-section');
            if (geminiSection.style.display === 'none' || geminiSection.style.display === '') {
                geminiSection.style.display = 'block';
            } else {
                geminiSection.style.display = 'none';
            }
        });

        document.getElementById('inventario').addEventListener('click', () => {
            agregarMensaje('Asistente: Muy bien! ¿En qué estás interesado?', 'asistente');
            // Hide the main chat options and show sub-options
            document.getElementById('chat-options').style.display = 'none';
            document.getElementById('sub-options').innerHTML = `
                <button class="sub-option-btn" id="buscar-libro">Buscar por libro</button>
                <button class="sub-option-btn" id="buscar-autor">Buscar por autor</button>
            `;
            document.getElementById('sub-options').style.display = 'block';
        });

    }

    // Sub-options handlers - consolidated into one listener
    document.addEventListener('click', (event) => {
        if (event.target.id === 'buscar-libro' || (event.target.classList.contains('sub-option-btn') && event.target.id === 'buscar-libro')) {
            console.log('Buscar libro clicked');
            currentSearchType = 'book';
            agregarMensaje('Asistente: Buscar por libro - Escribe el nombre del libro:', 'asistente');
            document.getElementById('sub-options').style.display = 'none';
            document.getElementById('search-input-container').style.display = 'block';
            document.getElementById('search-input').placeholder = 'Escribe el nombre del libro...';
            document.getElementById('search-input').focus();
        } else if (event.target.id === 'buscar-autor' || (event.target.classList.contains('sub-option-btn') && event.target.id === 'buscar-autor')) {
            console.log('Buscar autor clicked');
            currentSearchType = 'author';
            agregarMensaje('Asistente: Buscar por autor - Escribe el nombre del autor:', 'asistente');
            document.getElementById('sub-options').style.display = 'none';
            document.getElementById('search-input-container').style.display = 'block';
            document.getElementById('search-input').placeholder = 'Escribe el nombre del autor...';
            document.getElementById('search-input').focus();
        }
    });

    // Search functionality with autocomplete
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');

    searchInput.addEventListener('input', async () => {
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
    });

    searchBtn.addEventListener('click', async () => {
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
                    }
                } else {
                    agregarMensaje('Asistente: No se encontró una coincidencia exacta. Inténtalo de nuevo.', 'asistente');
                }
            } else {
                agregarMensaje('Asistente: No se encontraron resultados para tu búsqueda.', 'asistente');
            }
        } catch (error) {
            console.error('Error:', error);
            agregarMensaje('Asistente: Error al buscar. Inténtalo de nuevo.', 'asistente');
        }
    });

    // Handle item selection
    document.addEventListener('click', async (event) => {
        if (event.target.classList.contains('item-option-btn')) {
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
            agregarMensaje('Asistente: Usar como recomendación - Esta funcionalidad estará disponible próximamente.', 'asistente');
            resetChatOptions();
        } else if (event.target.id === 'informacion-autor') {
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
                    const booksList = data.books.map(book => `- ${book}`).join('\n');
                    agregarMensaje(`Asistente: Libros disponibles de ${selectedItem}:\n${booksList}`, 'asistente');
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
            // Clear chat and show initial options
            document.getElementById('chat-output').innerHTML = '<div class="asistente">¡Hola! Soy tu asistente de recomendación de lecturas. ¿En qué puedo ayudarte hoy?</div>';
            resetChatOptions();
            return;
        }
        // ... existing code for other actions
    });
});