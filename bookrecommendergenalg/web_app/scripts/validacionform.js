
function validacionForm() {
    // Obtener valores de los campos
    var nombre = document.getElementById('nombre').value.trim();
    var email = document.getElementById('email').value.trim();
    var telefono = document.getElementById('telefono').value.trim();
    var mensaje = document.getElementById('mensaje').value.trim();
    var terms = document.getElementById('terms').checked;

    
  
    var isValid = true;

   
    // Validar Nombre
    if (nombre === '') {
        document.getElementById('nombreError').style.display = 'block';
        isValid = false;
    } else {
        document.getElementById('nombreError').style.display = 'none';
    }

    // Validar Email
    if (email === '') {
        document.getElementById('emailError').style.display = 'block';
        isValid = false;
    } else {
        document.getElementById('emailError').style.display = 'none';
    }

    // Validar Teléfono
    if (telefono === '') {
        document.getElementById('errorTelefono').style.display = 'block';
        isValid = false;
    } else {
        document.getElementById('errorTelefono').style.display = 'none';
    }

    // Validar Mensaje
    if (mensaje === '') {
        document.getElementById('errorMensaje').style.display = 'block';
        isValid = false;
    } else {
        document.getElementById('errorMensaje').style.display = 'none';
    }

      // Validar Términos y condiciones
      if (!terms) {
        document.getElementById('termsError').style.display = 'block';
        isValid = false;
    } else {
        document.getElementById('termsError').style.display = 'none';
    }

    
}