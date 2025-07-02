package com.panda.oa.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Properties;

import javax.net.ssl.HttpsURLConnection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HttpUtils {
	
	public static Properties properties = new Properties();
	private final static Logger logger = LoggerFactory.getLogger(HttpUtils.class);
	
	public static void loadProperties() throws IOException {
		InputStream inputStream = HttpUtils.class.getClassLoader().getResourceAsStream("application.properties");
		InputStreamReader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);
		properties.load(reader);
		reader.close();
		inputStream.close();
	}
	
	public static String httpsRequest(String ipAddress, String method, Map<String, String> requestHeader, String requestBody) {
		StringBuilder response = new StringBuilder();
		try {
			URL url = new URL(ipAddress);
			HttpsURLConnection connection = (HttpsURLConnection) url.openConnection();
			connection.setHostnameVerifier((hostname, session) -> true);
			connection.setConnectTimeout(5000);
			connection.setReadTimeout(5000);
			connection.setRequestMethod(method);
			connection.setInstanceFollowRedirects(false);
			if(requestHeader != null) {
				for(Map.Entry<String, String> entry : requestHeader.entrySet()) {
					connection.setRequestProperty(entry.getKey(), entry.getValue());
				}
			}
			if(requestBody != null) {
				connection.setDoOutput(true);
				OutputStream outputStream = connection.getOutputStream();
				outputStream.write(requestBody.getBytes(StandardCharsets.UTF_8));
				outputStream.flush();
			}
			// 读取响应
	        BufferedReader readerResponse = new BufferedReader(new InputStreamReader(connection.getInputStream()));
	        String responseLine;
	        while ((responseLine = readerResponse.readLine()) != null) {
	            response.append(responseLine.trim());
	        }
		} catch (Exception e) {
			logger.error("网络请求出现异常，==》" + e.getMessage());
		}
		return response.toString();
	}
}