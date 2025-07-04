CREATE TABLE `Synth_text` (
  `id` int NOT NULL AUTO_INCREMENT,
  `label` varchar(50) DEFAULT NULL,
  `std_label` varchar(50) DEFAULT NULL,
  `pseudo_label` varchar(255) DEFAULT NULL,
  `oritxt` text,
  `cleanedtxt` text,
  `translate_chn` text,
  `translate_my` text,
  `translate_tm` text,
  `cm_en_chn` text,
  `cm_en_my` text,
  `cm_en_tm` text,
  `cm_chn_en` text,
  `cm_chn_my` text,
  `cm_chn_tm` text,
  `cm_my_en` text,
  `cm_my_chn` text,
  `cm_my_tm` text,
  `cm_tm_en` text,
  `cm_tm_chn` text,
  `cm_tm_my` text,
  `cw_en_chn` text,
  `cw_en_my` text,
  `cw_en_tm` text,
  `cw_chn_en` text,
  `cw_chn_my` text,
  `cw_chn_tm` text,
  `cw_my_en` text,
  `cw_my_chn` text,
  `cw_my_tm` text,
  `cw_tm_en` text,
  `cw_tm_chn` text,
  `cw_tm_my` text,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=981009 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
