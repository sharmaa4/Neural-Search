<?php
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Headers", "*");
var_dump($_FILES);

foreach ($_FILES["myFiles"]["tmp_name"] as $key => $value) {
	$targetPath = "uploads/" . basename($_FILES["myFiles"]["name"][$key]);
	move_uploaded_file($value, $targetPath);
}
