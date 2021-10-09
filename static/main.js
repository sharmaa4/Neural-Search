var Express = require('express');
 var multer = require('multer');
 var bodyParser = require('body-parser');
 var app = Express();
 app.use(bodyParser.json());


var Storage = multer.diskStorage({
     destination: function(req, file, callback) {
         callback(null, "./Images");
     },
     filename: function(req, file, callback) {
         callback(null, file.fieldname + "_" + Date.now() + "_" + file.originalname);
     }
 });

var upload = multer({
     storage: Storage
 }).array("myFiles", 3); //Field name and max count

app.post("/upload", (req, res) => {
     res.setHeader("Access-Control-Allow-Origin", "*")	 
     upload(req, res, function(err) {
	 res.setHeader("Access-Control-Allow-Origin", "*")
         if (err) {
             return res.end("Something went wrong!");
         }
         return res.end("File uploaded sucessfully!.");
     });
 });


app.listen(3000, () => {
	console.log("Server running!")

});
