package com.panda.oa.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.panda.oa.service.KPIService;

/**
 * @panda.chen
 * KPI绩效指标计算控制器
 */
@RestController
@RequestMapping("/kpi")
public class KPIController {
	@Autowired
	private KPIService kPIService;
	
	/**
	 * KPI绩效指标计算结果调用接口
	 * @param nowYear
	 * @return
	 */
	@GetMapping("/datas")
	public ResponseEntity<String> getKPIDatas(@RequestParam(name = "nowYear") String nowYear){
		if(nowYear == null) {
			return new ResponseEntity<String>(HttpStatus.NOT_FOUND);
		}
		return ResponseEntity.ok(kPIService.getKPIDatas(nowYear));
	}
	
}