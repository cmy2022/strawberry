package com.panda.oa.mapper;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import com.panda.oa.model.DataSource;
import com.panda.oa.model.DataSourceKey;

@Mapper
@DataSource(DataSourceKey.FIRST)
public interface KPIMapper {
	
	/**
	 * 根据编码查询IDs
	 * @param codes 编码集合
	 * @return
	 */
	@Select("<script>SELECT ID FROM table_name WHERE code IN (<foreach item='code' collection='codes' separator=','>#{code}</foreach>)</script>")
	public List<String> getOrganizationIdsByCodes(@Param("codes") List<String> codes);
	
}