const express = require('express');
const app = express();
const path = require('path');

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Define a route for handling the image upload
app.post('/upload', (req, res) => {
  // Handle the image upload logic here
  // You can access the uploaded image using req.body or req.files
  // Process the image as needed and send the response
});

// Start the server
const port = 3000; // You can use any port number you prefer
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
