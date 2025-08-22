// Existing FAQ functionality
document.querySelectorAll('.ques-content').forEach(item => {
    item.addEventListener('click', function() {
        // Toggle active class
        this.classList.toggle('active');
        
        // Toggle the visibility of the next sibling element (the answer)
        let answer = this.nextElementSibling;
        if (answer.style.display === "block") {
            answer.style.display = "none";
        } else {
            answer.style.display = "block";
        }
    });
});

// New functionality for the subscribe button
document.querySelector('.sub-btn').addEventListener('click', function() {
    // Get the email input field
    const emailField = document.querySelector('.email');
    
    // Check if the email field is not empty
    if (emailField.value.trim() !== "") {
        // Display a message
        const message = document.createElement('p');
        message.textContent = "You are now subscribed to the newsletter!";
        message.style.color = "#60535A"; // Add your desired color
        message.style.textAlign = 'center'; // Center align the text
        message.style.marginRight = '120px'; // Adjust the position
        message.style.marginTop = '20px'; // Add space above message
        
        // Append the message after the email input field
        document.querySelector('.newsletter').appendChild(message);
        
        // Clear the email input field
        emailField.value = '';
    } else {
        // If the email field is empty, show an error message
        const errorMessage = document.createElement('p');
        errorMessage.textContent = "Please enter a valid email address!";
        errorMessage.style.color = "red"; // Error message in red
        
        // Append the error message
        document.querySelector('.newsletter').appendChild(errorMessage);
    }
});

// Correct event listener for the "CHAT NOW" button
document.querySelector('.button').addEventListener('click', function() {
    window.location.href = 'chat.html'; // Redirect to chat.html
});

// Correct event listener for the "Get started" button
document.querySelector('.button1').addEventListener('click', function() {
    window.location.href = 'chat.html'; // Redirect to chat.html
});

