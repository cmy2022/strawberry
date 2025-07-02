package com.panda.oa.service;

import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import com.panda.oa.mapper.KPIMapper;
import com.panda.oa.model.KPIType;
import com.panda.oa.util.HttpUtils;

import lombok.Synchronized;

/**
 * @panda.chen
 * KPI绩效指标计算服务
 */
@Service
public class KPIService {
	
	@Autowired
	private KPIMapper kPIMapper;
	
	private final static Logger logger = LoggerFactory.getLogger(KPIService.class);
	private final static String REQUEST_BODY = "{\"Action\":\"getBaseKpi\"}";
	private final static String KPI_DATA = "";
	private final static Map<String, Method> methodMap = new HashMap<String, Method>();
	static {
		try {
			KPIType[] kPITypes = KPIType.values();
			for(int i=0; i<kPITypes.length; i++) {
				KPIType kPIType = kPITypes[i];
				methodMap.put(String.valueOf(kPIType.getCode()), KPIService.class.getMethod("computeKPI" + (i+1), String.class, String.class));
			}
		} catch (Exception e) {
			logger.error("KPI绩效指标编码和方法名称映射关系静态代码块执行出现异常，==》" + e.getMessage());
		}
	}
	
	/**
	 * （1）KPI绩效指标定时任务：每个月的月初执行一次;
	 */
	@Scheduled(cron = "0 0 0 1 * ?")
	public void scheduleTask() {
		getKPIDatas(null);
	}
	
	/**
	 * （2）同时作为API的处理逻辑提供前台调用。
	 * @param nowYear 当前需要计算KPI绩效指标的年度
	 * @return
	 */
	@Synchronized
	@Async("asyncExecutor")
	public String getKPIDatas(String thisYear) {
		JSONArray results = new JSONArray();
		JSONObject kPIBaseObject = new JSONObject(HttpUtils.httpsRequest(HttpUtils.properties.getProperty(KPI_DATA) + "?System=panda_kpi", "POST", null, REQUEST_BODY));
		logger.info("调用KPI绩效指标基础数据接口返回数据为：" + kPIBaseObject.toString());
		if(kPIBaseObject == null) {
			logger.error("调用KPI绩效指标基础数据接口发生错误,接口返回信息为空!");
		}else if (kPIBaseObject != null && !kPIBaseObject.getString("errorCode").equals("0")) {
			logger.error("调用KPI绩效指标基础数据接口发生错误，errorCode:{}, msg:{}", kPIBaseObject.getString("errorCode"), kPIBaseObject.getString("msg"));
		}else {
			logger.info("调用KPI绩效指标基础数据接口成功！");
		}
		JSONArray kPIBaseArray = kPIBaseObject.getJSONArray("data");
		Calendar calendar = Calendar.getInstance();
		int nowYear = calendar.get(Calendar.YEAR);
		int lastYear = nowYear - 1;
		int nowMonth = calendar.get(Calendar.MONTH) + 1;
		if(thisYear != null) {
			nowYear = Integer.valueOf(thisYear);
			lastYear = nowYear - 1;
			nowMonth = 12;
		}
		SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
		for(int i=0; i<kPIBaseArray.length(); i++) {
			JSONObject kPIBaseData = kPIBaseArray.getJSONObject(i);
			String period = kPIBaseData.getString("CYCLE");
			Double weight = Double.valueOf(kPIBaseData.getString("WEIGHT"));
			BigDecimal minTarget = new BigDecimal(kPIBaseData.getString("MIN_TARGET_VALUE"));
			BigDecimal kPITarget = new BigDecimal(kPIBaseData.getString("ASS_TARGET_VALUE"));
			BigDecimal highTarget = new BigDecimal(kPIBaseData.getString("CHA_TARGET_VALUE"));
			Method method = methodMap.get(kPIBaseData.getString("KPI_CODE"));
			for(int j=1; j<=nowMonth; j++) {
				try {
					calendar.setTime(dateFormat.parse(nowYear + "-" + j + "-01"));
					int maxDayOfMonth = calendar.getActualMaximum(Calendar.DAY_OF_MONTH);
					calendar.setTime(dateFormat.parse(lastYear + "-" + j + "-01"));
					int lastYearMaxDayOfMonth = calendar.getActualMaximum(Calendar.DAY_OF_MONTH);
					String lastYearValue = (String) method.invoke(this, new Object[]{lastYear + "-" + j + "-01", lastYear + "-" + j + "-" + lastYearMaxDayOfMonth});
					String value = (String) method.invoke(this, new Object[]{nowYear + "-" + j + "-01", nowYear + "-" + j + "-" + maxDayOfMonth});
					BigDecimal longValue = new BigDecimal(value);
					String score = null;
					String status = null;
					int sign = kPIBaseData.getString("DIRECTION").equals("正") ? -1 : 1;
					if(longValue.compareTo(minTarget) == sign) {
						score = "0";
						status = "未达标";
					}else if(longValue.compareTo(kPITarget) == sign) {
						score = String.valueOf(80 * weight);
						status = "未达标";
					}else if(longValue.compareTo(highTarget) == sign) {
						score = String.valueOf(100 * weight);
						status = "达标";
					}else {
						score = String.valueOf(120 * weight);
						status = "超额达成";
					}
					JSONObject result = postKPIResults(kPIBaseData, String.valueOf(nowYear), String.valueOf(j / 3 + (j % 3 == 0 ? 0 : 1)), nowYear + "-" + j, score, lastYearValue, value, status, null, null);
					results.put(result);
				} catch (Exception e) {
					logger.error("方法：【" + method.getName() + "】调用过程出现异常，==》" +  e.getMessage());
					e.printStackTrace();
				}
			}
			if(period.equals("季度")) {
				for(int j=0; j<4; j++) {
					JSONObject kPIResults = new JSONObject();
					
				}
			}else if(period.equals("年度")){
				JSONObject kPIResults = new JSONObject();
				
			}
		}
		return results.toString();
	}
	
	/**
	 * 调用KPI绩效指标保存接口
	 * @param kPIBaseData KPI基础数据
	 * @param year 当前考核年份
	 * @param quarter 当前考核季度
	 * @param month	当前考核月份
	 * @param score 得分
	 * @param lastYearValue 上年度同期值
	 * @param value 实际值
	 * @param status 达标情况
	 * @param passedNum 合格数量
	 * @param completedNum 完成数量
	 * @return
	 */
	public JSONObject postKPIResults(JSONObject kPIBaseData, String year, String quarter, String month, String score, String lastYearValue, String value, String status, String passedNum, String completedNum) {
		JSONObject kPIResults = new JSONObject();
		kPIResults.put("Action", "saveData");	//接口方法名称
		kPIResults.put("FUN_CENTER", kPIBaseData.getString("FUN_CENTER"));	//职能中心
		kPIResults.put("CHECK_UNIT", kPIBaseData.getString("CHECK_UNIT"));	//被考核单元
		kPIResults.put("CHECK_DEPT", kPIBaseData.getString("CHECK_DEPT"));	//被考核部门
		kPIResults.put("KPI_DESC", kPIBaseData.getString("KPI_DESC"));		//考核指标名称
		kPIResults.put("KPI_CODE", kPIBaseData.getString("KPI_CODE"));		//考核指标编码
		kPIResults.put("WEIGHT", kPIBaseData.getString("WEIGHT"));			//权重
		kPIResults.put("CYCLE", kPIBaseData.getString("CYCLE"));			//考核周期："月度"、"季度"、"年度"
		kPIResults.put("MIN_TARGET_VALUE", kPIBaseData.getString("MIN_TARGET_VALUE"));	//最低目标值
		kPIResults.put("ASS_TARGET_VALUE", kPIBaseData.getString("ASS_TARGET_VALUE"));	//考核目标值
		kPIResults.put("CHA_TARGET_VALUE", kPIBaseData.getString("CHA_TARGET_VALUE"));	//挑战目标值
		kPIResults.put("CHECK_YEAR", year);				//考核年度
		kPIResults.put("CHECK_QUA", quarter);			//考核季度
		kPIResults.put("CHECK_MONTH", month);			//考核月份
		kPIResults.put("ACTUAL_VALUE", value);			//实际值
		kPIResults.put("LAST_VALUE", lastYearValue);	//去年同期值
		kPIResults.put("SCORE", score);					//得分
		kPIResults.put("STATUS", status);				//达标情况
		kPIResults.put("PASS_NUM", passedNum);			//合格数量
		kPIResults.put("TOTAL_NUM", completedNum);		//完成数量
		
		JSONObject response = new JSONObject(HttpUtils.httpsRequest(HttpUtils.properties.getProperty(KPI_DATA) + "?System=panda_kpi", "POST", null, kPIResults.toString()));
		logger.info("调用KPI绩效指标【" + kPIBaseData.getString("KPI_DESC") + "】的保存接口请求参数为：" + kPIResults.toString());
		if(response == null) {
			logger.error("调用KPI绩效指标保存接口失败,接口返回信息为空!");
		}else if (response != null && !response.getString("errorCode").equals("0")){
			logger.error("调用KPI绩效指标保存接口失败, errorCode:{}, msg:{}", response.getString("errorCode"), response.getString("msg"));
		}else {
			logger.info("调用KPI绩效指标保存接口成功！");
		}
		return kPIResults;
	}
	
	/**
	 * KPI绩效指标1：【name1】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI1(String startDate, String endDate) {
		return null;
	}
	
	/**
	 * KPI绩效指标2：【name2】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI2(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标3：【name3】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI3(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标4：【name4】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI4(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标5：【name5】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI5(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标6：【name6】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI6(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标7：【name7】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI7(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标8：【name8】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI8(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标9：【name9】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI9(String startDate, String endDate) {
		return null;
	}

	/**
	 * KPI绩效指标10：【name10】
	 * @param startDate
	 * @param endDate
	 * @return
	 */
	public String computeKPI10(String startDate, String endDate) {
		return null;
	}
}