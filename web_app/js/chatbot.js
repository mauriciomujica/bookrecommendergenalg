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

    datasetForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent form submission

        const userId = document.getElementById('csv-search').value.trim();
        const selectionContainer = document.getElementById('dataset-selection');
        selectionContainer.innerHTML = ''; // clear previous selection UI

        if (!userId) {
            datasetResults.innerHTML = '<p>Please enter a User ID.</p>';
            return;
        }

        try {
            const response = await fetch(`/search-csv?user_id=${encodeURIComponent(userId)}`);
            if (response.ok) {
                const data = await response.json();
                if (data.results && Object.keys(data.results).length > 0) {
                    // Create checkbox list from object values (titles), with keys (ISBNs) as values
                    const results = Object.entries(data.results);
                    const listHtml = results.map(([isbn, title], idx) => {
                        return `<div class="checkbox-item"><label><input type="checkbox" data-index="${idx}" value="${escapeHtml(isbn)}"> ${escapeHtml(title)}</label></div>`;
                    }).join('');

                    datasetResults.innerHTML = `<div class="checkbox-list">${listHtml}</div>`;

                    // Add a button to get selected items
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.id = 'get-selected';
                    btn.textContent = 'Get selected';
                    btn.addEventListener('click', () => {
                        const checked = Array.from(datasetResults.querySelectorAll('input[type="checkbox"]:checked'));
                        const chosen = checked.map(cb => cb.value);
                        // Display chosen items in the selection container
                        if (chosen.length) {
                            selectionContainer.innerHTML = `<p>Selected (${chosen.length}):</p><ul>${chosen.map(x => `<li>${x}</li>`).join('')}</ul>`;
                        } else {
                            selectionContainer.innerHTML = '<p>No items selected.</p>';
                        }
                    });

                    // Add control buttons: Select all, Deselect all, Send Selected
                    const controlsHtml = document.createElement('div');
                    controlsHtml.className = 'checkbox-controls';

                    const selectAllBtn = document.createElement('button');
                    selectAllBtn.type = 'button';
                    selectAllBtn.id = 'select-all';
                    selectAllBtn.textContent = 'Select all';
                    selectAllBtn.addEventListener('click', () => {
                        datasetResults.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
                    });

                    const deselectAllBtn = document.createElement('button');
                    deselectAllBtn.type = 'button';
                    deselectAllBtn.id = 'deselect-all';
                    deselectAllBtn.textContent = 'Deselect all';
                    deselectAllBtn.addEventListener('click', () => {
                        datasetResults.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
                    });

                    const sendBtn = document.createElement('button');
                    sendBtn.type = 'button';
                    sendBtn.id = 'send-selected';
                    sendBtn.textContent = 'Send selected to server';
                    sendBtn.addEventListener('click', async () => {
                        const checked = Array.from(datasetResults.querySelectorAll('input[type="checkbox"]:checked'));
                        const chosen = checked.map(cb => cb.value);
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
                        }
                    });

                    controlsHtml.appendChild(selectAllBtn);
                    controlsHtml.appendChild(deselectAllBtn);
                    controlsHtml.appendChild(sendBtn);

                    // Append controls after the list, before the get-selected button
                    datasetResults.appendChild(controlsHtml);

                    // Append or replace existing button
                    const existingBtn = document.getElementById('get-selected');
                    if (existingBtn) existingBtn.replaceWith(btn); else datasetResults.appendChild(btn);

                } else {
                    datasetResults.innerHTML = '<p>No results found for the given User ID.</p>';
                }
            } else {
                datasetResults.innerHTML = '<p>Error searching the CSV file.</p>';
            }
        } catch (error) {
            console.error('Error:', error);
            datasetResults.innerHTML = '<p>Unable to connect to the server.</p>';
        }
    });

    // Function to display book recommendations with synopses in chatbot format
    function displayRecommendations(books, container) {
        let html = '<div class="recommendations-chat">';
        html += '<div class="asistente">¡Aquí tienes mis recomendaciones personalizadas basadas en tus gustos!</div>';

        books.forEach((book, index) => {
            const title = escapeHtml(book.title || `Libro ${index + 1}`);
            const synopsis = escapeHtml(book.synopsis || 'Sinopsis no disponible');

            html += `<div class="usuario"><strong>Recomendación ${index + 1}:</strong> ${title}</div>`;
            if (synopsis !== 'Sinopsis no disponible') {
                html += `<div class="asistente">${synopsis}</div>`;
            }
        });

        html += '<div class="asistente">¿Te gustaría más información sobre alguno de estos libros o nuevas recomendaciones?</div>';
        html += '</div>';

        // Add CSS styles for the chat-like appearance
        html += `
        <style>
            .recommendations-chat {
                max-width: 600px;
                margin: 20px auto;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .recommendations-chat > div {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
                line-height: 1.4;
            }
            .recommendations-chat .asistente {
                background-color: #e3f2fd;
                text-align: left;
                margin-left: 20px;
            }
            .recommendations-chat .usuario {
                background-color: #fff3e0;
                text-align: right;
                margin-right: 20px;
                font-weight: bold;
            }
        </style>`;

        container.innerHTML = html;
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