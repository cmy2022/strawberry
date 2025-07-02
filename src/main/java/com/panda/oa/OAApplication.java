package com.panda.oa;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import com.panda.oa.util.HttpUtils;


/**
 * @panda.chen
 * 主程序
 */
@SpringBootApplication
public class OAApplication implements CommandLineRunner{

	public static void main(String[] args) {
		SpringApplication.run(OAApplication.class, args);
	}
	
	@Override
	public void run(String... args) throws Exception {
		HttpUtils.loadProperties();
	}
}