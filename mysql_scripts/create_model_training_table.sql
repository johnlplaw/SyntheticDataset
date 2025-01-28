CREATE TABLE `model_training` (
  `model_name` varchar(255) NOT NULL,
  `epoch` int NOT NULL,
  `elapsed` double DEFAULT NULL,
  `training_loss` double DEFAULT NULL,
  `training_accuracy` double DEFAULT NULL,
  `val_loss` double DEFAULT NULL,
  `val_accuracy` double DEFAULT NULL,
  PRIMARY KEY (`model_name`,`epoch`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
