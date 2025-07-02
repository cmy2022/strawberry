package com.panda.oa.model;

/**
 * @author panda.chen
 *	KPI绩效指标
 */
public enum KPIType {
	
	KPIType1("name1", "code1"),
	KPIType2("name2", "code2"),
	KPIType3("name3", "code3"),
	KPIType4("name4", "code4"),
	KPIType5("name5", "code5"),
	KPIType6("name6", "code6"),
	KPIType7("name7", "code7"),
	KPIType8("name8", "code8"),
	KPIType9("name9", "code9"),
	KPIType10("name10", "code10");
	
	private final String name;
	private final String code;
	private KPIType(String name, String code){
		this.name = name;
		this.code = code;
	}
	public String getName(){
		return this.name;
	}
	public String getCode(){
		return this.code;
	}
}