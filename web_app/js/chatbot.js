document.addEventListener('DOMContentLoaded', () => {
    const mensajeInput = document.getElementById('mensaje');
    const enviarBoton = document.getElementById('enviarMensaje');
    const chatOutput = document.getElementById('chat-output');
    const datasetButton = document.getElementById('dataset-button');
    const chatbotButton = document.getElementById('chatbot-button');
    const datasetSection = document.getElementById('dataset-section');
    const chatbotSection = document.getElementById('chatbot-section');
    const datasetForm = document.getElementById('dataset-form');
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

    datasetForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent form submission

        const userId = document.getElementById('csv-search').value.trim();
        if (!userId) {
            datasetResults.innerHTML = '<p>Please enter a User ID.</p>';
            return;
        }

        await loadDataset(userId);
    });

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
});