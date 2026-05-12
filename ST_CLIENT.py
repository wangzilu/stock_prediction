# -*- coding: utf-8 -*-
# ST_CLIENT.py - StockToday API 客户端
# 支持所有 ST_SERVER.py 后台服务接口

import requests
import pandas as pd
import time
import random
from typing import Optional, Dict, Any, List


class StockToday:
    """StockToday API 客户端"""
    def __init__(self, base_url: str = "https://tushare.citydata.club/", token: str = "test", backup_url1: str = "http://111.229.164.2:8083/", backup_url2: str = "http://124.223.112.152:6331/", backup_url3: str = 'http://110.42.211.9:9900/'):
        """
        初始化客户端
        首次使用请替换token
        Args:
            base_url: 后台服务地址
            token: API Token
            backup_url: 备用请求地址
        """
        self.url = base_url
        self.backup_url1 = backup_url1
        self.backup_url2 = backup_url2
        self.backup_url3 = backup_url3
        self.TOKEN = token

    def _post(self, endpoint: str, params: Optional[Dict] = None, retry: int = 3) -> Any:
        """发送 POST 请求，负载均衡模式，随机选择服务器，重试时切换另一台"""
        if params is None:
            params = {}
        params['TOKEN'] = self.TOKEN
        urls = [self.url, self.backup_url1, self.backup_url2, self.backup_url3]
        tried = set()

        for attempt in range(retry):
            # 随机选择一台未尝试过的服务器
            available = [u for u in urls if u not in tried]
            if not available:
                tried.clear()  # 重置，已尝试完一轮
                available = urls
            url = random.choice(available)
            tried.add(url)

            headers = {}
            try:
                resp = requests.post(f"{url}{endpoint}", data=params, headers=headers, timeout=10)
                if resp.status_code == 200 and resp.text.strip():
                    return resp.json()
                # 非200或空响应，切换服务器重试
            except requests.exceptions.Timeout:
                pass  # 超时直接切换
            except Exception:
                pass  # 其他错误直接切换

        return {"error": f"请求失败，已重试{retry}次"}



    # ==================== 2.1 基础数据 ====================

    def stock_basic(self, exchange: str = "", list_status: str = "L", fields: str = "",
                    ts_code: str = "", market: str = "") -> Any:
        """
        股票基本信息
        """
        params = {"exchange": exchange, "list_status": list_status, "fields": fields}
        if ts_code:
            params["ts_code"] = ts_code
        if market:
            params["market"] = market
        return self._post("/stock_basic", params)

    def stk_premarket(self, ts_code: str = "", trade_date: str = "",
                      start_date: str = "", end_date: str = "") -> Any:
        """新股上市"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_premarket", params)

    def trade_cal(self, exchange: str = "SSE", start_date: str = "", end_date: str = "", is_open: str = "") -> Any:
        """
        交易日历
        """
        params = {"exchange": exchange}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if is_open:
            params["is_open"] = is_open
        return self._post("/trade_cal", params)
        
    def stock_st(self, ts_code: str = "",trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """
        ST股票列表
        """
        params = {"trade_date": trade_date}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stock_st", params)

    def st(self, ts_code: str = "", pub_date: str = "", imp_date: str = "") -> Any:
        """
        ST风险警示板股票
        """
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if pub_date:
            params["pub_date"] = pub_date
        if imp_date:
            params["imp_date"] = imp_date
        return self._post("/st", params)

    def namechange(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """股票名称变更"""
        params = {"ts_code": ts_code}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/namechange", params)

    def stock_company(self, ts_code: str = "", exchange: str = "") -> Any:
        """上市公司信息"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if exchange:
            params["exchange"] = exchange
        return self._post("/stock_company", params)

    def stk_managers(self, ts_code: str = "", start_date: str = "", end_date: str = "",
                     ann_date: str = "") -> Any:
        """公司管理层"""
        params = {"ts_code": ts_code}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if ann_date:
            params["ann_date"] = ann_date
        return self._post("/stk_managers", params)

    def stk_rewards(self, ts_code: str = "") -> Any:
        """高管薪酬"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/stk_rewards", params)

    def new_share(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """新股上市"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/new_share", params)

    def bak_basic(self, trade_date: str = "", ts_code: str = "") -> Any:
        """备用基础数据（股票历史列表）"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/bak_basic", params)

    # ==================== 2.2 行情数据 ====================

    def daily(self, ts_code: str = "", trade_date: str = "",
              start_date: str = "", end_date: str = "") -> Any:
        """
        日线行情
        """
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/daily", params)

    def weekly(self, ts_code: str = "", trade_date: str = "",
               start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """周线行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/weekly", params)

    def monthly(self, ts_code: str = "", trade_date: str = "",
                start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """月线行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/monthly", params)

    def pro_bar(self, ts_code: str, start_date: str = "", end_date: str = "",
                asset: str = "E", adj: str = "qfq", freq: str = "D",
                ma: str = "", factors: str = "", adjfactor: str = "") -> Any:
        """
        行情数据（支持复权/定时务）
        """
        params = {
            "ts_code": ts_code,
            "asset": asset,
            "adj": adj,
            "freq": freq
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if ma:
            params["ma"] = ma
        if factors:
            params["factors"] = factors
        if adjfactor:
            params["adjfactor"] = adjfactor
        return self._post("/pro_bar", params)

    def adj_factor(self, ts_code: str = "", trade_date: str = "",
                   start_date: str = "", end_date: str = "") -> Any:
        """复权因子"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/adj_factor", params)

    def daily_basic(self, ts_code: str = "", trade_date: str = "",
                    start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """每日指标"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/daily_basic", params)

    def stk_limit(self, ts_code: str = "", trade_date: str = "",
                  start_date: str = "", end_date: str = "") -> Any:
        """涨跌停"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_limit", params)

    def suspend_d(self, ts_code: str = "", trade_date: str = "",
                  start_date: str = "", end_date: str = "", suspend_type: str = "1") -> Any:
        """停复牌信息"""
        params = {"suspend_type": suspend_type}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/suspend_d", params)

    def hsgt_top10(self, trade_date: str = "", start_date: str = "",
                   end_date: str = "", market_type: str = "") -> Any:
        """沪深股通前十成交"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if market_type:
            params["market_type"] = market_type
        return self._post("/hsgt_top10", params)

    def ggt_top10(self, trade_date: str = "", ts_code: str = "",
                  start_date: str = "", end_date: str = "") -> Any:
        """广港通前十成交"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/ggt_top10", params)

    def ggt_daily(self, trade_date: str = "",
                  start_date: str = "", end_date: str = "") -> Any:
        """广港通每日成交"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/ggt_daily", params)

    def ggt_monthly(self, trade_date: str = "",
                    start_date: str = "", end_date: str = "") -> Any:
        """广港通每月成交"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/ggt_monthly", params)

    def bak_daily(self, ts_code: str = "", trade_date: str = "",
                  start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """备用每日行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/bak_daily", params)

    # ==================== 2.3 财务数据 ====================

    def income(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", f_ann_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """利润表"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if f_ann_date:
            params["f_ann_date"] = f_ann_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/income", params)

    def balancesheet(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """资产负债表"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/balancesheet", params)

    def cashflow(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", f_ann_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """现金流量表"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if f_ann_date:
            params["f_ann_date"] = f_ann_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/cashflow", params)

    def forecast(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", type: str = "", fields: str = "") -> Any:
        """业绩预告"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if type:
            params["type"] = type
        if fields:
            params["fields"] = fields
        return self._post("/forecast", params)

    def express(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", fields: str = "") -> Any:
        """业绩快报"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if fields:
            params["fields"] = fields
        return self._post("/express", params)

    def dividend(self, ts_code: str = "", ann_date: str = "", record_date: str = "", ex_date: str = "", imp_ann_date: str = "", fields: str = "") -> Any:
        """分红送股"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if record_date:
            params["record_date"] = record_date
        if ex_date:
            params["ex_date"] = ex_date
        if imp_ann_date:
            params["imp_ann_date"] = imp_ann_date
        if fields:
            params["fields"] = fields
        return self._post("/dividend", params)

    # ============ VIP接口 ============
    def income_vip(self, ts_code: str, ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """VIP利润表"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/income_vip", params)

    def balancesheet_vip(self, ts_code: str, ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """VIP资产负债表"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/balancesheet_vip", params)

    def cashflow_vip(self, ts_code: str, ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """VIP现金流量表"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/cashflow_vip", params)

    def fina_indicator_vip(self, ts_code: str, ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", report_type: str = "", comp_type: str = "", fields: str = "") -> Any:
        """VIP财务指标"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if comp_type:
            params["comp_type"] = comp_type
        if fields:
            params["fields"] = fields
        return self._post("/fina_indicator_vip", params)

    def express_vip(self, ts_code: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """VIP业绩快报"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/express_vip", params)

    def forecast_vip(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", type: str = "", fields: str = "") -> Any:
        """VIP业绩预告"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if type:
            params["type"] = type
        if fields:
            params["fields"] = fields
        return self._post("/forecast_vip", params)

    def fina_mainbz_vip(self, ts_code: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """VIP主营业务构成"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/fina_mainbz_vip", params)

    def fina_indicator(self, ts_code: str, ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "", fields: str = "") -> Any:
        """财务指标"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        if fields:
            params["fields"] = fields
        return self._post("/fina_indicator", params)

    def fina_audit(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", period: str = "") -> Any:
        """财务审计意见"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if period:
            params["period"] = period
        return self._post("/fina_audit", params)

    def fina_mainbz(self, ts_code: str = "", type: str = "", period: str = "", start_date: str = "", end_date: str = "") -> Any:
        """主营业务"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if type:
            params["type"] = type
        if period:
            params["period"] = period
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/fina_mainbz", params)

    def disclosure_date(self, ts_code: str = "", end_date: str = "", start_date: str = "", period: str = "") -> Any:
        """披露日期"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if end_date:
            params["end_date"] = end_date
        if start_date:
            params["start_date"] = start_date
        if period:
            params["period"] = period
        return self._post("/disclosure_date", params)

    # ==================== 2.4 参考数据 ====================

    def top10_holders(self, ts_code: str, ann_date: str = "", period: str = "", start_date: str = "", end_date: str = "") -> Any:
        """十大股东"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if period:
            params["period"] = period
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/top10_holders", params)

    def top10_floatholders(self, ts_code: str, ann_date: str = "", period: str = "", start_date: str = "", end_date: str = "") -> Any:
        """十大流通股东"""
        params = {"ts_code": ts_code}
        if ann_date:
            params["ann_date"] = ann_date
        if period:
            params["period"] = period
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/top10_floatholders", params)

    def pledge_stat(self, ts_code: str = "", end_date: str = "") -> Any:
        """股权质押统计"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if end_date:
            params["end_date"] = end_date
        return self._post("/pledge_stat", params)

    def pledge_detail(self, ts_code: str = "") -> Any:
        """股权质押明细"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/pledge_detail", params)

    def repurchase(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """股份回购"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/repurchase", params)

    def share_float(self, ts_code: str = "", ann_date: str = "", float_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """限售股解禁"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if float_date:
            params["float_date"] = float_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/share_float", params)

    def block_trade(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """大宗交易"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/block_trade", params)

    def stk_holdernumber(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """股东户数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_holdernumber", params)

    def stk_holdertrade(self, ts_code: str = "", ann_date: str = "", start_date: str = "", end_date: str = "", trade_type: str = "", holder_type: str = "") -> Any:
        """股东增减持"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if trade_type:
            params["trade_type"] = trade_type
        if holder_type:
            params["holder_type"] = holder_type
        return self._post("/stk_holdertrade", params)

    # ==================== 2.5 特色数据 ====================

    def report_rc(self, ts_code: str = "", report_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """研报"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if report_date:
            params["report_date"] = report_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/report_rc", params)

    def cyq_perf(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """筹码活跃度"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/cyq_perf", params)

    def cyq_chips(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """筹码分布"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/cyq_chips", params)

    def stk_factor_pro(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """股票因子(专业版)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/stk_factor_pro", params)

    def ccass_hold(self, ts_code: str = "", start_date: str = "", end_date: str = "", trade_date: str = "", hk_code: str = "") -> Any:
        """中央结算持股"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if trade_date:
            params["trade_date"] = trade_date
        if hk_code:
            params["hk_code"] = hk_code
        return self._post("/ccass_hold", params)

    def ccass_hold_detail(self, ts_code: str = "", trade_date: str = "", fields: str = "") -> Any:
        """中央结算持股明细"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if fields:
            params["fields"] = fields
        return self._post("/ccass_hold_detail", params)

    def hk_hold(self, ts_code: str = "", code: str = "", exchange: str = "", start_date: str = "", end_date: str = "", trade_date: str = "") -> Any:
        """港股持股"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if code:
            params["code"] = code
        if exchange:
            params["exchange"] = exchange
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/hk_hold", params)

    def stk_auction_o(self, ts_code: str = "", trade_date: str = "",
                      start_date: str = "", end_date: str = "") -> Any:
        """集合竞价"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_auction_o", params)

    def stk_auction_c(self, ts_code: str = "", trade_date: str = "",
                      start_date: str = "", end_date: str = "") -> Any:
        """盘后定价"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_auction_c", params)

    def stk_nineturn(self, ts_code: str = "", trade_date: str = "",
                     start_date: str = "", end_date: str = "", freq: str = "", fields: str = "") -> Any:
        """九转序列"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        if fields:
            params["fields"] = fields
        return self._post("/stk_nineturn", params)

    def stk_surv(self, ts_code: str = "", trade_date: str = "", fields: str = "") -> Any:
        """舆情监控"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if fields:
            params["fields"] = fields
        return self._post("/stk_surv", params)

    def broker_recommend(self, month: str) -> Any:
        """券商月度金股"""
        params = {"month": month}
        return self._post("/broker_recommend", params)

    # ==================== 2.6 两融数据 ====================

    def margin(self, trade_date: str = "") -> Any:
        """融资融券"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/margin", params)

    def margin_detail(self, trade_date: str = "") -> Any:
        """融资融券明细"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/margin_detail", params)

    def margin_secs(self, trade_date: str = "", exchange: str = "") -> Any:
        """融资融券证券"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if exchange:
            params["exchange"] = exchange
        return self._post("/margin_secs", params)

    def slb_sec(self, trade_date: str = "") -> Any:
        """融券余量"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/slb_sec", params)

    def slb_len(self, trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """融资期限"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/slb_len", params)

    def slb_sec_detail(self, trade_date: str = "") -> Any:
        """融券余量明细"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/slb_sec_detail", params)

    def slb_len_mm(self, trade_date: str = "") -> Any:
        """融资期限明细"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/slb_len_mm", params)

    # ==================== 2.7 资金流向 ====================

    def moneyflow(self, ts_code: str = "", trade_date: str = "",
                  start_date: str = "", end_date: str = "") -> Any:
        """资金流向"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow", params)

    def moneyflow_ths(self, ts_code: str = "", trade_date: str = "",
                      start_date: str = "", end_date: str = "") -> Any:
        """资金流向(同花顺)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_ths", params)

    def moneyflow_cnt_ths(self, trade_date: str = "", ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """资金流向分类(同花顺)"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_cnt_ths", params)

    def moneyflow_dc(self, ts_code: str = "", trade_date: str = "",
                     start_date: str = "", end_date: str = "") -> Any:
        """资金流向(东方财富)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_dc", params)

    def moneyflow_ind_ths(self, ts_code: str = "", trade_date: str = "",
                          start_date: str = "", end_date: str = "") -> Any:
        """行业资金流向(同花顺)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_ind_ths", params)

    def moneyflow_ind_dc(self, ts_code: str = "", trade_date: str = "",
                         start_date: str = "", end_date: str = "") -> Any:
        """行业资金流向(东方财富)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_ind_dc", params)

    def moneyflow_mkt_dc(self, trade_date: str = "",
                         start_date: str = "", end_date: str = "") -> Any:
        """市场资金流向(东方财富)"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_mkt_dc", params)

    def moneyflow_hsgt(self, ts_code: str = "", trade_date: str = "",
                       start_date: str = "", end_date: str = "") -> Any:
        """沪深港通资金流向"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/moneyflow_hsgt", params)

    # ==================== 2.8 打板专题 ====================

    def kpl_concept(self, trade_date: str = "") -> Any:
        """开盘啦概念"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/kpl_concept", params)

    def kpl_concept_cons(self, trade_date: str = "", ts_code: str = "", con_code: str = "") -> Any:
        """开盘啦概念成分"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if con_code:
            params["con_code"] = con_code
        return self._post("/kpl_concept_cons", params)

    def kpl_list(self, ts_code: str = "", trade_date: str = "", tag: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """开盘啦列表"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if tag:
            params["tag"] = tag
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/kpl_list", params)

    def top_list(self, trade_date: str = "", ts_code: str = "") -> Any:
        """龙虎榜-上榜"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/top_list", params)

    def top_inst(self, trade_date: str = "", ts_code: str = "") -> Any:
        """龙虎榜-机构"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/top_inst", params)

    def limit_list_ths(self, trade_date: str = "", ts_code: str = "", limit_type: str = "", market: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """涨停列表（同花顺）"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if limit_type:
            params["limit_type"] = limit_type
        if market:
            params["market"] = market
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/limit_list_ths", params)

    def limit_list_d(self, trade_date: str = "", ts_code: str = "", limit_type: str = "", exchange: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """涨跌停明细"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if limit_type:
            params["limit_type"] = limit_type
        if exchange:
            params["exchange"] = exchange
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/limit_list_d", params)

    def limit_step(self, ts_code: str = "", trade_date: str = "",
                   start_date: str = "", end_date: str = "", nums: str = "") -> Any:
        """涨停阶梯"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if nums:
            params["nums"] = nums
        return self._post("/limit_step", params)

    def limit_cpt_list(self, ts_code: str = "", trade_date: str = "",
                       start_date: str = "", end_date: str = "") -> Any:
        """涨停股票池"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/limit_cpt_list", params)

    def ths_index(self, ts_code: str = "", exchange: str = "", type: str = "", name: str = "") -> Any:
        """同花顺指数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if exchange:
            params["exchange"] = exchange
        if type:
            params["type"] = type
        if name:
            params["name"] = name
        return self._post("/ths_index", params)

    def ths_member(self, ts_code: str = "", con_code: str = "", start_date: str = "", end_date: str = "", is_new: str = "") -> Any:
        """同花顺成分股"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if con_code:
            params["con_code"] = con_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if is_new:
            params["is_new"] = is_new
        return self._post("/ths_member", params)

    def dc_index(self, ts_code: str = "", name: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", idx_type: str = "概念板块") -> Any:
        """东财指数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if name:
            params["name"] = name
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if idx_type:
            params["idx_type"] = idx_type
        return self._post("/dc_index", params)

    def dc_member(self, ts_code: str = "", con_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """东财成分股（必传trade_date）"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if con_code:
            params["con_code"] = con_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/dc_member", params)

    def stk_auction(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """股票集合竞价"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_auction", params)

    def hm_list(self, name: str = "") -> Any:
        """活跃股列表"""
        params = {}
        if name:
            params["name"] = name
        return self._post("/hm_list", params)

    def hm_detail(self, trade_date: str = "", ts_code: str = "", hm_name: str = "", start_date: str = "", end_date: str = "") -> Any:
        """活跃股明细"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if hm_name:
            params["hm_name"] = hm_name
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hm_detail", params)

    def ths_hot(self, market: str = "", trade_date: str = "", ts_code: str = "", is_new: str = "Y", fields: str = "") -> Any:
        """同花顺热榜"""
        params = {}
        if market:
            params["market"] = market
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if is_new:
            params["is_new"] = is_new
        if fields:
            params["fields"] = fields
        return self._post("/ths_hot", params)

    def dc_hot(self, market: str = "", hot_type: str = "", trade_date: str = "", is_new: str = "", ts_code: str = "", fields: str = "") -> Any:
        """东方财富热点"""
        params = {}
        if market:
            params["market"] = market
        if hot_type:
            params["hot_type"] = hot_type
        if trade_date:
            params["trade_date"] = trade_date
        if is_new:
            params["is_new"] = is_new
        if ts_code:
            params["ts_code"] = ts_code
        if fields:
            params["fields"] = fields
        return self._post("/dc_hot", params)

    def dc_concept(self, trade_date: str = "", theme_code: str = "", name: str = "", fields: str = "") -> Any:
        """东方财富题材库"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if theme_code:
            params["theme_code"] = theme_code
        if name:
            params["name"] = name
        if fields:
            params["fields"] = fields
        return self._post("/dc_concept", params)

    def dc_concept_cons(self, ts_code: str = "", trade_date: str = "", theme_code: str = "", fields: str = "") -> Any:
        """东方财富题材成分"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if theme_code:
            params["theme_code"] = theme_code
        if fields:
            params["fields"] = fields
        return self._post("/dc_concept_cons", params)

    # ==================== 3. 指数专题 ====================

    def index_basic(self, market: str = "", ts_code: str = "", publisher: str = "", category: str = "", symbol: str = "", name: str = "") -> Any:
        """指数基本信息"""
        params = {}
        if market:
            params["market"] = market
        if ts_code:
            params["ts_code"] = ts_code
        if publisher:
            params["publisher"] = publisher
        if category:
            params["category"] = category
        if symbol:
            params["symbol"] = symbol
        if name:
            params["name"] = name
        return self._post("/index_basic", params)

    def index_daily(self, ts_code: str, start_date: str = "", end_date: str = "") -> Any:
        """指数日线"""
        params = {"ts_code": ts_code}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/index_daily", params)

    def index_weekly(self, ts_code: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """指数周线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/index_weekly", params)

    def index_monthly(self, ts_code: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """指数月线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/index_monthly", params)

    def index_weight(self, index_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """指数成分"""
        params = {}
        if index_code:
            params["index_code"] = index_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/index_weight", params)

    def index_dailybasic(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """指数每日指标"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/index_dailybasic", params)

    def index_classify(self, level: str = "", src: str = "") -> Any:
        """指数分类"""
        params = {}
        if level:
            params["level"] = level
        if src:
            params["src"] = src
        return self._post("/index_classify", params)

    def index_member_all(self, l1_code: str = "", l2_code: str = "", l3_code: str = "", ts_code: str = "") -> Any:
        """指数成分股(全)"""
        params = {}
        if l1_code:
            params["l1_code"] = l1_code
        if l2_code:
            params["l2_code"] = l2_code
        if l3_code:
            params["l3_code"] = l3_code
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/index_member_all", params)

    def daily_info(self, trade_date: str = "", exchange: str = "", fields: str = "") -> Any:
        """每日信息"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if exchange:
            params["exchange"] = exchange
        if fields:
            params["fields"] = fields
        return self._post("/daily_info", params)

    def sz_daily_info(self, trade_date: str = "", ts_code: str = "") -> Any:
        """深市每日信息"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/sz_daily_info", params)

    def ths_daily(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """同花顺指数日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/ths_daily", params)

    def ci_daily(self, trade_date: str = "", fields: str = "") -> Any:
        """中证指数日线"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if fields:
            params["fields"] = fields
        return self._post("/ci_daily", params)

    def sw_daily(self, trade_date: str = "", fields: str = "") -> Any:
        """申万指数日线"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if fields:
            params["fields"] = fields
        return self._post("/sw_daily", params)

    def index_global(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """全球指数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/index_global", params)

    def idx_factor_pro(self, ts_code: str = "", trade_date: str = "",
                        start_date: str = "", end_date: str = "") -> Any:
        """指数因子(专业版)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/idx_factor_pro", params)

    # ==================== 4. 公募基金 ====================

    def fund_basic(self, market: str = "", status: str = "", ts_code: str = "") -> Any:
        """基金基本信息"""
        params = {}
        if market:
            params["market"] = market
        if status:
            params["status"] = status
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/fund_basic", params)

    def fund_company(self) -> Any:
        """基金公司"""
        return self._post("/fund_company")

    def fund_manager(self, ts_code: str = "") -> Any:
        """基金经理"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/fund_manager", params)

    def fund_share(self, ts_code: str = "") -> Any:
        """基金份额"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/fund_share", params)

    def fund_nav(self, ts_code: str = "") -> Any:
        """基金净值"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/fund_nav", params)

    def fund_div(self, ts_code: str = "", ann_date: str = "") -> Any:
        """基金分红"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if ann_date:
            params["ann_date"] = ann_date
        return self._post("/fund_div", params)

    def fund_portfolio(self, ts_code: str = "") -> Any:
        """基金持仓"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/fund_portfolio", params)

    def fund_daily(self, ts_code: str, trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """基金日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/fund_daily", params)

    def fund_adj(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """基金复权"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/fund_adj", params)

    def fund_factor_pro(self, ts_code: str = "", trade_date: str = "",
                         start_date: str = "", end_date: str = "") -> Any:
        """基金因子(专业版)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/fund_factor_pro", params)

    # ==================== 5. 期货数据 ====================

    def fut_basic(self, exchange: str = "", fut_type: str = "", list_date: str = "", fut_code: str = "", ts_code: str = "", fields: str = "") -> Any:
        """期货基本信息"""
        params = {}
        if exchange:
            params["exchange"] = exchange
        if fut_type:
            params["fut_type"] = fut_type
        if list_date:
            params["list_date"] = list_date
        if fut_code:
            params["fut_code"] = fut_code
        if ts_code:
            params["ts_code"] = ts_code
        if fields:
            params["fields"] = fields
        return self._post("/fut_basic", params)

    def fut_daily(self, ts_code: str = "", trade_date: str = "",
                  start_date: str = "", end_date: str = "", exchange: str = "", fields: str = "") -> Any:
        """期货日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if exchange:
            params["exchange"] = exchange
        if fields:
            params["fields"] = fields
        return self._post("/fut_daily", params)

    def fut_weekly_monthly(self, ts_code: str = "", trade_date: str = "",
                           start_date: str = "", end_date: str = "", freq: str = "", exchange: str = "") -> Any:
        """期货周/月线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        if exchange:
            params["exchange"] = exchange
        return self._post("/fut_weekly_monthly", params)

    def ft_mins(self, ts_code: str = "", start_date: str = "", end_date: str = "", freq: str = "") -> Any:
        """期货分钟线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/ft_mins", params)

    def fut_wsr(self, trade_date: str = "", symbol: str = "") -> Any:
        """期货持仓"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if symbol:
            params["symbol"] = symbol
        return self._post("/fut_wsr", params)

    def fut_settle(self, trade_date: str = "", exchange: str = "") -> Any:
        """期货结算"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if exchange:
            params["exchange"] = exchange
        return self._post("/fut_settle", params)

    def fut_holding(self, trade_date: str = "", symbol: str = "", exchange: str = "") -> Any:
        """期货持仓量"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if symbol:
            params["symbol"] = symbol
        if exchange:
            params["exchange"] = exchange
        return self._post("/fut_holding", params)

    def fut_mapping(self, ts_code: str = "", trade_date: str = "",
                    start_date: str = "", end_date: str = "") -> Any:
        """期货映射"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/fut_mapping", params)

    def fut_weekly_detail(self, prd: str = "", start_week: str = "", end_week: str = "", fields: str = "") -> Any:
        """期货每周详情"""
        params = {}
        if prd:
            params["prd"] = prd
        if start_week:
            params["start_week"] = start_week
        if end_week:
            params["end_week"] = end_week
        if fields:
            params["fields"] = fields
        return self._post("/fut_weekly_detail", params)

    def ft_limit(self, trade_date: str = "", ts_code: str = "", cont: str = "") -> Any:
        """期货涨跌停"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if cont:
            params["cont"] = cont
        return self._post("/ft_limit", params)

    # ==================== 6. 现货数据 ====================

    def sge_basic(self, ts_code: str = "") -> Any:
        """现货基本信息"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/sge_basic", params)

    def sge_daily(self, trade_date: str = "", prd: str = "",
                  start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """现货每日行情"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if prd:
            params["prd"] = prd
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/sge_daily", params)

    # ==================== 7. 期权数据 ====================

    def opt_basic(self, exchange: str = "", fields: str = "") -> Any:
        """期权基本信息"""
        params = {}
        if exchange:
            params["exchange"] = exchange
        if fields:
            params["fields"] = fields
        return self._post("/opt_basic", params)

    def opt_daily(self, ts_code: str = "", trade_date: str = "",
                  start_date: str = "", end_date: str = "", exchange: str = "") -> Any:
        """期权每日行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if exchange:
            params["exchange"] = exchange
        return self._post("/opt_daily", params)

    # ==================== 8. 可转债/债券数据 ====================

    def cb_basic(self, fields: str = "") -> Any:
        """可转债基本信息"""
        params = {}
        if fields:
            params["fields"] = fields
        return self._post("/cb_basic", params)

    def cb_issue(self, ann_date: str = "", fields: str = "") -> Any:
        """可转债发行"""
        params = {}
        if ann_date:
            params["ann_date"] = ann_date
        if fields:
            params["fields"] = fields
        return self._post("/cb_issue", params)

    def cb_call(self, fields: str = "") -> Any:
        """可转债回售"""
        params = {}
        if fields:
            params["fields"] = fields
        return self._post("/cb_call", params)

    def cb_rate(self, ts_code: str = "", fields: str = "") -> Any:
        """可转债转股溢价率"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if fields:
            params["fields"] = fields
        return self._post("/cb_rate", params)

    def cb_daily(self, trade_date: str = "", ts_code: str = "",
                 start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """可转债每日行情"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/cb_daily", params)

    def cb_price_chg(self, ts_code: str = "", fields: str = "") -> Any:
        """可转债价格变化"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if fields:
            params["fields"] = fields
        return self._post("/cb_price_chg", params)

    def cb_share(self, ts_code: str = "", fields: str = "") -> Any:
        """可转债转股"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if fields:
            params["fields"] = fields
        return self._post("/cb_share", params)

    def cb_factor_pro(self, ts_code: str = "", trade_date: str = "",
                      start_date: str = "", end_date: str = "") -> Any:
        """可转债因子(专业版)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/cb_factor_pro", params)

    def repo_daily(self, trade_date: str = "") -> Any:
        """回购每日行情"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/repo_daily", params)

    def bc_otcqt(self, ts_code: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """银行间报价"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/bc_otcqt", params)

    def bc_bestotcqt(self, ts_code: str = "", start_date: str = "", end_date: str = "", fields: str = "") -> Any:
        """银行间最优报价"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        return self._post("/bc_bestotcqt", params)

    def bond_blk(self, start_date: str = "", end_date: str = "") -> Any:
        """债券大宗交易"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/bond_blk", params)

    def bond_blk_detail(self, start_date: str = "", end_date: str = "") -> Any:
        """债券大宗交易明细"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/bond_blk_detail", params)

    def yc_cb(self, trade_date: str = "", curve_type: str = "") -> Any:
        """可转债收益率"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if curve_type:
            params["curve_type"] = curve_type
        return self._post("/yc_cb", params)

    # ==================== 9. 宏观经济 ====================

    def eco_cal(self, date: str = "", country: str = "", event: str = "", fields: str = "") -> Any:
        """经济日历"""
        params = {}
        if date:
            params["date"] = date
        if country:
            params["country"] = country
        if event:
            params["event"] = event
        if fields:
            params["fields"] = fields
        return self._post("/eco_cal", params)

    def shibor(self, date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """Shibor利率"""
        params = {}
        if date:
            params["date"] = date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/shibor", params)

    def shibor_quote(self, bank: str = "", date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """Shibor报价"""
        params = {}
        if bank:
            params["bank"] = bank
        if date:
            params["date"] = date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/shibor_quote", params)

    def cn_gdp(self, q: str = "", start_q: str = "", end_q: str = "", fields: str = "") -> Any:
        """中国GDP"""
        params = {}
        if q:
            params["q"] = q
        if start_q:
            params["start_q"] = start_q
        if end_q:
            params["end_q"] = end_q
        if fields:
            params["fields"] = fields
        return self._post("/cn_gdp", params)

    def cn_cpi(self, m: str = "", start_m: str = "", end_m: str = "") -> Any:
        """中国CPI"""
        params = {}
        if m:
            params["m"] = m
        if start_m:
            params["start_m"] = start_m
        if end_m:
            params["end_m"] = end_m
        return self._post("/cn_cpi", params)

    def cn_ppi(self, m: str = "", start_m: str = "", end_m: str = "") -> Any:
        """中国PPI"""
        params = {}
        if m:
            params["m"] = m
        if start_m:
            params["start_m"] = start_m
        if end_m:
            params["end_m"] = end_m
        return self._post("/cn_ppi", params)

    def sf_month(self, m: str = "", start_m: str = "", end_m: str = "") -> Any:
        """上海黄金现货月报"""
        params = {}
        if m:
            params["m"] = m
        if start_m:
            params["start_m"] = start_m
        if end_m:
            params["end_m"] = end_m
        return self._post("/sf_month", params)

    # ==================== 10. 外汇数据 ====================

    def fx_obasic(self, exchange: str = "", classify: str = "", fields: str = "") -> Any:
        """外汇基本信息"""
        params = {}
        if exchange:
            params["exchange"] = exchange
        if classify:
            params["classify"] = classify
        if fields:
            params["fields"] = fields
        return self._post("/fx_obasic", params)

    def fx_daily(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """外汇每日行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/fx_daily", params)

    # ==================== 11. 港股数据 ====================

    def hk_basic(self, list_status: str = "", trade_date: str = "") -> Any:
        """港股基本信息"""
        params = {}
        if list_status:
            params["list_status"] = list_status
        if trade_date:
            params["trade_date"] = trade_date
        return self._post("/hk_basic", params)

    def hk_tradecal(self, start_date: str = "", end_date: str = "", is_open: str = "") -> Any:
        """港股交易日历"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if is_open:
            params["is_open"] = is_open
        return self._post("/hk_tradecal", params)

    def hk_daily(self, ts_code: str = "", trade_date: str = "",
                 start_date: str = "", end_date: str = "") -> Any:
        """港股每日行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_daily", params)

    def hk_daily_adj(self, ts_code: str = "", trade_date: str = "",
                     start_date: str = "", end_date: str = "") -> Any:
        """港股每日行情(复权)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_daily_adj", params)

    def hk_mins(self, ts_code: str, start_date: str = "", end_date: str = "", freq: str = "") -> Any:
        """港股分钟线"""
        params = {"ts_code": ts_code}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/hk_mins", params)

    def hk_income(self, ts_code: str, period: str = "", ind_name: str = "", start_date: str = "", end_date: str = "") -> Any:
        """港股利润表"""
        params = {"ts_code": ts_code}
        if period:
            params["period"] = period
        if ind_name:
            params["ind_name"] = ind_name
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_income", params)

    def hk_balancesheet(self, ts_code: str, period: str = "", ind_name: str = "", start_date: str = "", end_date: str = "") -> Any:
        """港股资产负债表"""
        params = {"ts_code": ts_code}
        if period:
            params["period"] = period
        if ind_name:
            params["ind_name"] = ind_name
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_balancesheet", params)

    def hk_cashflow(self, ts_code: str, period: str = "", ind_name: str = "", start_date: str = "", end_date: str = "") -> Any:
        """港股现金流量表"""
        params = {"ts_code": ts_code}
        if period:
            params["period"] = period
        if ind_name:
            params["ind_name"] = ind_name
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_cashflow", params)

    # ==================== 12. 美股数据 ====================

    def us_basic(self) -> Any:
        """美股基本信息"""
        return self._post("/us_basic")

    def us_tradecal(self, start_date: str = "", end_date: str = "", is_open: str = "") -> Any:
        """美股交易日历"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if is_open:
            params["is_open"] = is_open
        return self._post("/us_tradecal", params)

    def us_daily(self, ts_code: str = "", trade_date: str = "",
                 start_date: str = "", end_date: str = "") -> Any:
        """美股每日行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/us_daily", params)

    def us_daily_adj(self, ts_code: str = "", trade_date: str = "",
                     start_date: str = "", end_date: str = "", exchange: str = "") -> Any:
        """美股每日行情(复权)"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if exchange:
            params["exchange"] = exchange
        return self._post("/us_daily_adj", params)

    def us_income(self, ts_code: str, period: str = "", ind_name: str = "", report_type: str = "", start_date: str = "", end_date: str = "") -> Any:
        """美股利润表"""
        params = {"ts_code": ts_code}
        if period:
            params["period"] = period
        if ind_name:
            params["ind_name"] = ind_name
        if report_type:
            params["report_type"] = report_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/us_income", params)

    def us_balancesheet(self, ts_code: str, period: str = "", ind_name: str = "", report_type: str = "", start_date: str = "", end_date: str = "") -> Any:
        """美股资产负债表"""
        params = {"ts_code": ts_code}
        if period:
            params["period"] = period
        if ind_name:
            params["ind_name"] = ind_name
        if report_type:
            params["report_type"] = report_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/us_balancesheet", params)

    def us_cashflow(self, ts_code: str, period: str = "", ind_name: str = "", report_type: str = "", start_date: str = "", end_date: str = "") -> Any:
        """美股现金流量表"""
        params = {"ts_code": ts_code}
        if period:
            params["period"] = period
        if ind_name:
            params["ind_name"] = ind_name
        if report_type:
            params["report_type"] = report_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/us_cashflow", params)

    # ==================== 13. ETF/ETF数据 ====================

    def etf_basic(self, ts_code: str = "", index_code: str = "", list_date: str = "",
                  list_status: str = "", exchange: str = "", mgr_name: str = "") -> Any:
        """ETF基本信息
        ts_code: 证券代码
        index_code: 指数代码
        list_date: 上市日期
        list_status: 上市状态（L上市 D退市 P待上市）
        exchange: 交易所（SH上交所 SZ深交所）
        mgr_name: 基金管理人简称
        """
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if index_code:
            params["index_code"] = index_code
        if list_date:
            params["list_date"] = list_date
        if list_status:
            params["list_status"] = list_status
        if exchange:
            params["exchange"] = exchange
        if mgr_name:
            params["mgr_name"] = mgr_name
        return self._post("/etf_basic", params)

    def etf_index(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """ETF指数跟踪"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/etf_index", params)

    def etf_share_size(self, ts_code: str = "") -> Any:
        """ETF份额变化"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/etf_share_size", params)

    # ==================== 14. 特色指数 ====================

    def dc_daily(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", idx_type: str = "") -> Any:
        """东财指数日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if idx_type:
            params["idx_type"] = idx_type
        return self._post("/dc_daily", params)

    def gz_index(self, ts_code: str = "", trade_date: str = "",
                 start_date: str = "", end_date: str = "") -> Any:
        """国证指数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/gz_index", params)

    def wz_index(self, ts_code: str = "", trade_date: str = "",
                 start_date: str = "", end_date: str = "") -> Any:
        """万德指数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/wz_index", params)

    def tdx_index(self, ts_code: str = "", trade_date: str = "", idx_type: str = "") -> Any:
        """通达信指数"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if idx_type:
            params["idx_type"] = idx_type
        return self._post("/tdx_index", params)

    def tdx_daily(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """通达信指数日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/tdx_daily", params)

    # ==================== 15. 资讯数据 ====================

    def news(self, src: str = "", start_date: str = "", end_date: str = "") -> Any:
        """资讯"""
        params = {}
        if src:
            params["src"] = src
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/news", params)

    def major_news(self, trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """重要资讯"""
        params = {}
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/major_news", params)

    # ==================== 16. 基金销售 ====================

    def fund_sales_ratio(self, year: str = "") -> Any:
        """基金销售比例"""
        params = {}
        if year:
            params["year"] = year
        return self._post("/fund_sales_ratio", params)

    def fund_sales_vol(self, year: str = "", quarter: str = "", name: str = "") -> Any:
        """基金销售量"""
        params = {}
        if year:
            params["year"] = year
        if quarter:
            params["quarter"] = quarter
        if name:
            params["name"] = name
        return self._post("/fund_sales_vol", params)

    # ==================== 17. 其他高价值数据 ====================

    def stk_account(self, start_date: str = "", end_date: str = "") -> Any:
        """新增开户数"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_account", params)

    def stk_ah_comparison(self, hk_code: str = "", ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """AH股比价"""
        params = {}
        if hk_code:
            params["hk_code"] = hk_code
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_ah_comparison", params)

    def ci_index_member(self, index_code: str = "", ts_code: str = "") -> Any:
        """中证指数成分股权"""
        params = {}
        if index_code:
            params["index_code"] = index_code
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/ci_index_member", params)

    # ==================== 补充缺失接口 ====================

    def anns_d(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """上市公司公告"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/anns_d", params)

    def cctv_news(self, date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """新闻联播文字稿"""
        params = {}
        if date:
            params["date"] = date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/cctv_news", params)

    def ths_news(self, start_date: str = "", end_date: str = "", trade_date: str = "", limit: int = 100) -> Any:
        """同花顺新闻"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if trade_date:
            params["trade_date"] = trade_date
        if limit:
            params["limit"] = limit
        return self._post("/ths_news", params)

    def npr(self, start_date: str = "", end_date: str = "", search: str = "") -> Any:
        """国家政策库"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if search:
            params["search"] = search
        return self._post("/npr", params)

    def research_report(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """券商研究报告"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/research_report", params)

    def stock_hsgt(self, ts_code: str = "", trade_date: str = "", type: str = "", start_date: str = "", end_date: str = "") -> Any:
        """沪深港通股票列表"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if type:
            params["type"] = type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stock_hsgt", params)

    def bse_mapping(self, ts_code: str = "") -> Any:
        """北交所新旧代码对照"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/bse_mapping", params)

    def stk_account_old(self, start_date: str = "", end_date: str = "") -> Any:
        """股票开户数据(旧)"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/stk_account_old", params)

    def stk_weekly_monthly(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "week") -> Any:
        """周月线行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/stk_weekly_monthly", params)

    def stk_week_month_adj(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "week") -> Any:
        """周月线复权行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/stk_week_month_adj", params)

    def realtime_list(self) -> Any:
        """实时行情列表"""
        return self._post("/realtime_list", {})

    def realtime_quote(self, ts_code: str = "") -> Any:
        """实时报价"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/realtime_quote", params)

    def realtime_tick(self, ts_code: str = "") -> Any:
        """实时分笔成交"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/realtime_tick", params)

    def rt_min(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """股票实时分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/rt_min", params)

    def rt_etf_min(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """ETF实时分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/rt_etf_min", params)

    def rt_k(self, ts_code: str = "", asset: str = "E") -> Any:
        """股票实时日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if asset:
            params["asset"] = asset
        return self._post("/rt_k", params)

    def rt_idx_k(self, ts_code: str = "") -> Any:
        """指数实时日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/rt_idx_k", params)

    def rt_idx_min(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """指数实时分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/rt_idx_min", params)

    def rt_sw_k(self, ts_code: str = "") -> Any:
        """申万实时行情"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/rt_sw_k", params)

    def rt_etf_k(self, ts_code: str = "", topic: str = "") -> Any:
        """ETF实时日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if topic:
            params["topic"] = topic
        return self._post("/rt_etf_k", params)

    def rt_fut_min(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """期货实时分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/rt_fut_min", params)

    def rt_hk_k(self, ts_code: str = "") -> Any:
        """港股实时日线"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        return self._post("/rt_hk_k", params)

    def stk_mins(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """股票历史分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/stk_mins", params)

    def etf_mins(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """ETF历史分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/etf_mins", params)

    def idx_mins(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """指数历史分钟"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/idx_mins", params)

    def opt_mins(self, ts_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "", freq: str = "1min") -> Any:
        """期权分钟行情（start_date/end_date需datetime格式如'2024-08-25 09:00:00'）"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if freq:
            params["freq"] = freq
        return self._post("/opt_mins", params)

    def hk_adjfactor(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """港股复权因子"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_adjfactor", params)

    def hk_fina_indicator(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """港股财务指标"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hk_fina_indicator", params)

    def us_adjfactor(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """美股复权因子"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/us_adjfactor", params)

    def us_fina_indicator(self, ts_code: str = "", period: str = "", report_type: str = "", start_date: str = "", end_date: str = "") -> Any:
        """美股财务指标"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if period:
            params["period"] = period
        if report_type:
            params["report_type"] = report_type
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/us_fina_indicator", params)

    def us_tbr(self, start_date: str = "", end_date: str = "", fields: str = "", date: str = "") -> Any:
        """美国短期国债利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        if date:
            params["date"] = date
        return self._post("/us_tbr", params)

    def us_tycr(self, start_date: str = "", end_date: str = "", fields: str = "", date: str = "") -> Any:
        """美国国债收益率曲线利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        if date:
            params["date"] = date
        return self._post("/us_tycr", params)

    def us_trycr(self, start_date: str = "", end_date: str = "", fields: str = "", date: str = "") -> Any:
        """美国国债实际收益率曲线利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        if date:
            params["date"] = date
        return self._post("/us_trycr", params)

    def us_tltr(self, start_date: str = "", end_date: str = "", fields: str = "", date: str = "") -> Any:
        """美国国债长期利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        if date:
            params["date"] = date
        return self._post("/us_tltr", params)

    def us_trltr(self, start_date: str = "", end_date: str = "", fields: str = "", date: str = "") -> Any:
        """美国国债实际长期利率平均值"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if fields:
            params["fields"] = fields
        if date:
            params["date"] = date
        return self._post("/us_trltr", params)

    def libor(self, start_date: str = "", end_date: str = "", curr_type: str = "", date: str = "") -> Any:
        """Libor利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if curr_type:
            params["curr_type"] = curr_type
        if date:
            params["date"] = date
        return self._post("/libor", params)

    def hibor(self, start_date: str = "", end_date: str = "") -> Any:
        """Hibor利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/hibor", params)

    def shibor_lpr(self, start_date: str = "", end_date: str = "") -> Any:
        """LPR贷款基础利率"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/shibor_lpr", params)

    def cn_m(self, start_date: str = "", end_date: str = "") -> Any:
        """货币供应量(月)"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/cn_m", params)

    def cn_pmi(self, start_date: str = "", end_date: str = "") -> Any:
        """采购经理指数PMI"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/cn_pmi", params)

    def tdx_member(self, ts_code: str = "", con_code: str = "", trade_date: str = "", start_date: str = "", end_date: str = "") -> Any:
        """通达信板块成分"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if con_code:
            params["con_code"] = con_code
        if trade_date:
            params["trade_date"] = trade_date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/tdx_member", params)

    def tmt_twincome(self, start_date: str = "", end_date: str = "") -> Any:
        """台湾电子产业月营收"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/tmt_twincome", params)

    def tmt_twincomedetail(self, start_date: str = "", end_date: str = "") -> Any:
        """台湾电子产业月营收明细"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/tmt_twincomedetail", params)

    def film_record(self, start_date: str = "", end_date: str = "") -> Any:
        """电影剧本备案公示"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/film_record", params)

    def teleplay_record(self, start_date: str = "", end_date: str = "") -> Any:
        """电视剧备案公示"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/teleplay_record", params)

    def bo_daily(self, date: str = "") -> Any:
        """电影日度票房"""
        params = {}
        if date:
            params["date"] = date
        return self._post("/bo_daily", params)

    def bo_monthly(self, date: str = "") -> Any:
        """电影月度票房"""
        params = {}
        if date:
            params["date"] = date
        return self._post("/bo_monthly", params)

    def bo_weekly(self, date: str = "") -> Any:
        """电影周度票房"""
        params = {}
        if date:
            params["date"] = date
        return self._post("/bo_weekly", params)

    def bo_cinema(self, date: str = "") -> Any:
        """影院日度票房"""
        params = {}
        if date:
            params["date"] = date
        return self._post("/bo_cinema", params)

    def irm_qa_sh(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """上证e互动问答"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/irm_qa_sh", params)

    def irm_qa_sz(self, ts_code: str = "", start_date: str = "", end_date: str = "") -> Any:
        """深证易互动问答"""
        params = {}
        if ts_code:
            params["ts_code"] = ts_code
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._post("/irm_qa_sz", params)


# ==================== 便捷函数 ====================

def get_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取日线数据"""
    st = StockToday()
    data = st.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame()


def get_index_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取指数日线"""
    st = StockToday()
    data = st.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame()


def get_realtime_quote(ts_code: str) -> Dict:
    """获取实时行情（使用 pro_bar）"""
    st = StockToday()
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y%m%d")
    return st.pro_bar(ts_code=ts_code, start_date=today, end_date=today)


if __name__ == "__main__":
    st = StockToday()

    # ==================== 2.1 基础数据 ====================
    # 股票基本信息 - 可按市场/状态/代码筛选
    # print(st.stock_basic())                              # 全部上市股票
    # print(st.stock_basic(list_status="L"))               # 仅上市
    # print(st.stock_basic(list_status="P"))               # 暂停上市(空)
    # print(st.stock_basic(market="科创板"))                # 609条
    # print(st.stock_basic(ts_code="000001.SZ"))            # 指定股票

    # 交易日历
    # print(st.trade_cal(start_date="20260101", end_date="20260110"))  # 10条
    # print(st.trade_cal(exchange="SSE", is_open="1"))

    # 日线行情
    # print(st.daily(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))  # 5条
    # print(st.daily(trade_date="20260506"))                # 5460条

    # 通用行情(pro_bar) - 可查股票/指数/期货/期权
    # print(st.pro_bar(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))  # 5条
    # print(st.pro_bar(ts_code="000001.SH", asset="I"))    # 指数 8000条
    # print(st.pro_bar(ts_code="IF2506.CFX", asset="FT"))  # 期货
    # print(st.pro_bar(ts_code="10004883.SH", asset="O"))  # 期权

    # 周月线 - freq参数必传，week/month (积分需要2000+)
    # print(st.stk_weekly_monthly(ts_code="000001.SZ", start_date="20250101", end_date="20251231", freq="week"))  # 周线
    # print(st.stk_weekly_monthly(ts_code="000001.SZ", start_date="20250101", end_date="20251231", freq="month"))  # 月线
    # print(st.stk_week_month_adj(ts_code="000001.SZ", start_date="20250101", end_date="20251231", freq="week"))  # 复权周线
    # 周线 - trade_date为每周最后一个交易日
    # print(st.stk_weekly_monthly(trade_date="20251024", freq="week"))
    # 月线 - trade_date为每月最后一个交易日
    # print(st.stk_weekly_monthly(trade_date="20251031", freq="month"))

    # 指数信息
    # print(st.index_basic(market="SSE"))                  # 594条
    # print(st.index_daily(ts_code="000001.SH", start_date="20260101", end_date="20260110"))  # 5条
    # print(st.index_weight(index_code="000001.SH", start_date="20250101", end_date="20251231"))  # 6000条

    # 基金信息
    # print(st.fund_basic(market="E"))                      # 2560条
    # print(st.fund_daily(ts_code="000001.OF", trade_date="20260428"))  # 1960条

    # 期货信息
    # print(st.fut_basic(exchange="CFFEX"))                # 700条
    # print(st.fut_daily(ts_code="IF2506.CFX", start_date="20250101", end_date="20251231"))  # 111条

    # 宏观经济
    # print(st.cn_pmi(start_date="20260101", end_date="20260506"))  # 255条
    # print(st.cn_m(start_date="20260101", end_date="20260506"))    # 579条
    # print(st.shibor(start_date="20260101", end_date="20260110"))   # 6条
    # print(st.shibor_lpr(start_date="20260101", end_date="20260506"))  # 3条

    # 热点数据
    # print(st.dc_hot(market="A股市场"))                    # 2000条
    # print(st.dc_hot(is_new="1"))                         # 2000条
    # print(st.ths_hot(market="热股"))                     # 2000条

    # 券商金股
    # print(st.broker_recommend(month="202504"))           # 124条

    # print(st.stock_company(ts_code="000001.SZ"))
    # print(st.stk_managers(ts_code="000001.SZ"))
    # print(st.stk_rewards(ts_code="000001.SZ"))
    # print(st.new_share(start_date="20260101", end_date="20260110"))
    # print(st.bak_basic(trade_date="20260110"))
    # print(st.bak_basic(ts_code="300605.SZ"))  0

    # ==================== 2.2 行情数据 ====================
    # print(st.daily(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.daily(trade_date="20260430"))
    # print(st.weekly(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.monthly(ts_code="000001.SZ", start_date="20250101", end_date="20250110"))
    # print(st.pro_bar(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.adj_factor(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.daily_basic(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.stk_limit(ts_code="000001.SZ", trade_date="20260110"))
    # print(st.suspend_d(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.hsgt_top10(trade_date="20260110"))
    # print(st.ggt_top10(trade_date="20260110"))
    # print(st.ggt_daily(trade_date="20260110"))
    # print(st.ggt_monthly(trade_date="202601"))
    # print(st.bak_daily(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))

    # ==================== 2.3 财务数据 ====================
    # print(st.income(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))     # 利润表 2条
    # print(st.balancesheet(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))  # 资产负债表 2条
    # print(st.cashflow(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))   # 现金流量表 3条
    # print(st.forecast(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))   # 业绩预告(空)
    # print(st.express(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))    # 业绩快报(空)
    # print(st.dividend(ts_code="000001.SZ"))                                               # 分红 51条
    # print(st.fina_indicator(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))  # 财务指标 3条
    # print(st.fina_audit(ts_code="000001.SZ"))                                           # 审计意见 39条
    # print(st.fina_mainbz(ts_code="000001.SZ"))                                          # 主营构成 150条
    # print(st.disclosure_date(ts_code="000001.SZ", start_date="20250101", end_date="20250630"))  # 披露日期 5720条

    # ==================== 2.4 参考数据 ====================
    # print(st.top10_holders(ts_code="000001.SZ"))
    # print(st.top10_floatholders(ts_code="000001.SZ"))
    # print(st.pledge_stat(ts_code="000001.SZ"))
    # print(st.pledge_detail(ts_code="000001.SZ"))
    # print(st.repurchase(ts_code="000001.SZ"))
    # print(st.share_float(ts_code="000001.SZ"))
    # print(st.block_trade(trade_date="20260428"))
    # print(st.stk_holdernumber(ts_code="000001.SZ"))
    # print(st.stk_holdertrade(ts_code="000001.SZ"))

    # ==================== 2.5 特色数据 ====================
    # print(st.report_rc(ts_code="000001.SZ"))
    # print(st.cyq_perf(ts_code="000001.SZ", trade_date="20260428"))
    # print(st.cyq_chips(ts_code="000001.SZ"))
    # print(st.stk_factor_pro(ts_code="000001.SZ", start_date="20260401", end_date="20260428"))
    # print(st.ccass_hold(ts_code="00700.HK"))
    # print(st.ccass_hold_detail(ts_code="00700.HK", trade_date="20260428"))
    # print(st.hk_hold(ts_code="00700.HK"))
    # print(st.stk_auction_o())
    # print(st.stk_auction_c())
    # print(st.stk_nineturn(ts_code="000001.SZ"))
    # print(st.stk_surv(ts_code="000001.SZ"))
    # print(st.broker_recommend(month="202604"))

    # ==================== 2.6 两融数据 ====================
    # print(st.margin(trade_date="20260110"))
    # print(st.margin_detail(trade_date="20260110", ts_code="000001.SZ"))
    # print(st.margin_secs(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.slb_sec(ts_code="000001.SZ"))
    # print(st.slb_len(trade_date="20260110"))
    # print(st.slb_sec_detail(ts_code="000001.SZ", trade_date="20260110"))
    # print(st.slb_len_mm(ts_code="000001.SZ", month="202601"))

    # ==================== 2.7 资金流向 ====================
    # print(st.moneyflow(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_ths(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_cnt_ths(start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_dc(market="SH", start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_ind_ths(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_ind_dc(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_mkt_dc(start_date="20260101", end_date="20260110"))
    # print(st.moneyflow_hsgt(trade_date="20260110"))

    # ==================== 2.8 打板专题 ====================
    # print(st.kpl_concept())
    # print(st.kpl_concept_cons(id="KPL_C18", ts_code="000001.SZ"))
    # print(st.kpl_list(trade_date="20260110"))
    # print(st.top_list(trade_date="20260110"))
    # print(st.top_inst(trade_date="20260110"))
    # print(st.limit_list_ths(trade_date="20260110"))
    # print(st.limit_list_d(trade_date="20260110"))
    # print(st.limit_step(trade_date="20260110"))
    # print(st.limit_cpt_list(trade_date="20260110"))
    # print(st.ths_index(ts_code="885521.TI"))
    # print(st.ths_member(ts_code="885521.TI"))
    # print(st.dc_index(ts_code="000001.SH"))
    # print(st.dc_member(ts_code="000001.SH"))
    # print(st.stk_auction(trade_date="20260430"))
    # print(st.hm_list())
    # print(st.hm_detail(trade_date="20260428"))
    # print(st.ths_hot(market="热股"))
    # print(st.dc_hot(market="A股市场"))
    # print(st.dc_concept(trade_date="20260505"))
    # print(st.dc_concept_cons(trade_date="20260505"))

    # ==================== 3. 指数专题 ====================
    # print(st.index_basic(market="SSE"))
    # print(st.index_daily(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.index_weekly(ts_code="000001.SH", start_date="20250101", end_date="20250110"))
    # print(st.index_monthly(ts_code="000001.SH", start_date="20250101", end_date="20250110"))
    # print(st.index_weight(index_code="000001.SH", start_date="20260401", end_date="20260428"))
    # print(st.index_dailybasic(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.index_classify(level="L1"))
    # print(st.index_member_all(index_code="000001.SH"))
    # print(st.daily_info(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.sz_daily_info(trade_date="20260110"))
    # print(st.ths_daily(start_date="20260401", end_date="20260428"))
    # print(st.ci_daily(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.sw_daily(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.index_global(ts_code="SPX"))
    # print(st.idx_factor_pro(ts_code="000001.SH", start_date="20260101", end_date="20260110"))

    # ==================== 4. 公募基金 ====================
    # print(st.fund_basic())
    # print(st.fund_company())
    # print(st.fund_manager(ts_code="000001.OF"))
    # print(st.fund_share(ts_code="000001.OF"))
    # print(st.fund_nav(ts_code="000001.OF"))
    # print(st.fund_div(ts_code="000001.OF"))
    # print(st.fund_portfolio(ts_code="000001.OF"))
    # print(st.fund_daily(ts_code="000001.OF", trade_date="20260428"))
    # print(st.fund_adj(ts_code="000001.OF"))
    # print(st.fund_factor_pro(ts_code="000001.OF", start_date="20260101", end_date="20260110"))

    # ==================== 5. 期货数据 ====================
    # print(st.fut_basic(exchange="CFFEX"))
    # print(st.fut_daily(ts_code="IF2506.CFX", start_date="20260101", end_date="20260110"))
    # print(st.fut_weekly_monthly(ts_code="IF2506.CFX", start_date="20260101", end_date="20260110"))
    # print(st.ft_mins(ts_code="IF2506.CFX"))
    # print(st.fut_wsr(trade_date="20260428"))
    # print(st.fut_settle(ts_code="IF2506.CFX", trade_date="20260428"))
    # print(st.fut_holding(ts_code="IF2506.CFX", trade_date="20260428"))
    # print(st.fut_mapping())
    # print(st.fut_weekly_detail())
    # print(st.ft_limit(ts_code="IF2506.CFX", trade_date="20260428"))

    # ==================== 6. 现货数据 ====================
    # print(st.sge_basic())
    # print(st.sge_daily(ts_code="Au99.99", start_date="20260101", end_date="20260110"))

    # ==================== 7. 期权数据 ====================
    # print(st.opt_basic())
    # print(st.opt_daily(ts_code="10004883.SH", start_date="20260101", end_date="20260110"))

    # ==================== 8. 可转债/债券数据 ====================
    # print(st.cb_basic())
    # print(st.cb_issue(ann_date="20260401"))
    # print(st.cb_call())
    # print(st.cb_rate(ts_code="113009.SH"))
    # print(st.cb_daily(ts_code="113009.SH", start_date="20260101", end_date="20260110"))
    # print(st.cb_price_chg())
    # print(st.cb_share(ts_code="113009.SH"))
    # print(st.cb_factor_pro(ts_code="113009.SH"))
    # print(st.repo_daily(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.bc_otcqt(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.bc_bestotcqt(ts_code="000001.SH", start_date="20260101", end_date="20260110"))
    # print(st.bond_blk())
    # print(st.bond_blk_detail(ts_code="000001.SH"))
    # print(st.yc_cb())

    # ==================== 9. 宏观经济 ====================
    # print(st.shibor(start_date="20260101", end_date="20260110"))       # 6条
    # print(st.shibor_lpr(start_date="20260101", end_date="20260506"))   # 3条
    # print(st.cn_gdp(q="2025Q1"))                                      # GDP
    # print(st.cn_cpi(start_date="202601", end_date="202612"))            # CPI
    # print(st.cn_ppi(start_date="202601", end_date="202612"))           # PPI
    # print(st.cn_pmi(start_date="20260101", end_date="20260506"))        # PMI 255条
    # print(st.cn_m(start_date="20260101", end_date="20260506"))          # 货币供应量 579条
    # print(st.libor(start_date="20180101", end_date="20181130"))         # Libor利率 4000条
    # print(st.libor(curr_type="USD", start_date="20180101", end_date="20181130"))  # Libor指定货币
    # print(st.hibor(start_date="20180101", end_date="20181130"))        # Hibor利率 227条

    # ==================== 10. 外汇数据 ====================
    # print(st.fx_obasic())                                             # 外汇基础
    # print(st.fx_daily(ts_code="USD/CNY", start_date="20260101", end_date="20260110"))

    # ==================== 10. 外汇数据 ====================
    # print(st.fx_obasic())
    # print(st.fx_daily(ts_code="USD/CNY", start_date="20260101", end_date="20260110"))

    # ==================== 11. 港股数据 ====================
    # print(st.hk_basic())
    # print(st.hk_tradecal(start_date="20260101", end_date="20261231"))
    # print(st.hk_daily(ts_code="00700.HK"))
    # print(st.hk_daily_adj(ts_code="00700.HK"))
    # print(st.hk_mins(ts_code="00700.HK"))
    # print(st.hk_income(ts_code="00700.HK"))
    # print(st.hk_balancesheet(ts_code="00700.HK"))
    # print(st.hk_cashflow(ts_code="00700.HK"))

    # ==================== 12. 美股数据 ====================
    # print(st.us_basic())
    # print(st.us_tradecal(start_date="20260101", end_date="20261231"))
    # print(st.us_daily(ts_code="AAPL"))
    # print(st.us_daily_adj(ts_code="AAPL"))
    # print(st.us_income(ts_code="AAPL"))
    # print(st.us_balancesheet(ts_code="AAPL"))
    # print(st.us_cashflow(ts_code="AAPL"))

    # ==================== 13. ETF/ETF数据 ====================
    # print(st.etf_basic())
    # print(st.etf_index(ts_code="510300"))
    # print(st.etf_share_size(ts_code="510300"))

    # ==================== 14. 特色指数 ====================
    # print(st.dc_daily(ts_code="000001.SH", start_date="20260401", end_date="20260428"))
    # print(st.gz_index())
    # print(st.wz_index())
    # print(st.tdx_index())
    # print(st.tdx_daily(ts_code="001004", start_date="20260401", end_date="20260428"))

    # ==================== 15. 资讯数据 ====================
    # print(st.news())
    # print(st.major_news())

    # ==================== 16. 基金销售 ====================
    # print(st.fund_sales_ratio(ts_code="000001.OF"))
    # print(st.fund_sales_vol(ts_code="000001.OF"))

    # ==================== 17. 其他高价值数据 ====================
    # print(st.stk_account(start_date="20260401", end_date="20260428"))
    # print(st.stk_ah_comparison(ts_code="000001.SZ"))
    # print(st.ci_index_member(index_code="000001.SH"))

    # ==================== 补充缺失接口 ====================
    # print(st.anns_d(ts_code="000001.SZ"))
    # print(st.cctv_news())
    # print(st.ths_news(start_date="20260101", end_date="20260506"))    # 同花顺新闻
    # print(st.npr())                                                  # 国家政策库(空)
    # print(st.research_report())                                       # 券商研究报告(空)
    # print(st.stock_hsgt(ts_code="000001.SZ"))
    # print(st.bse_mapping(ts_code="000001.SZ"))
    # print(st.stk_account_old(start_date="20260401", end_date="20260428"))
    # print(st.stk_weekly_monthly(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.stk_week_month_adj(ts_code="000001.SZ", start_date="20260101", end_date="20260110"))
    # print(st.realtime_list())                                         # 实时行情列表
    # print(st.realtime_quote(ts_code="000001.SZ"))                      # 实时报价
    # print(st.realtime_tick(ts_code="000001.SZ"))                       # 实时分笔成交
    # print(st.rt_min(ts_code="000001.SZ"))
    # print(st.rt_idx_k(ts_code="000001.SH", ktype="D"))
    # print(st.rt_idx_min(ts_code="000001.SH"))
    # print(st.rt_sw_k(ts_code="850511.SI"))                            # 申万实时行情
    # print(st.rt_etf_k(ts_code="510300.SH"))                           # ETF实时日线(需单独权限)
    # print(st.rt_etf_k(ts_code="5*.SH", topic="HQ_FND_TICK"))       # 沪市ETF需传topic
    # print(st.rt_etf_min(ts_code="510300.SH"))                         # ETF实时分钟
    # print(st.rt_fut_min(ts_code="IF2506.CFX"))                        # 期货实时分钟
    # print(st.rt_hk_k(ts_code="00700.HK"))                             # 港股实时日线
    print(st.stk_mins(ts_code="000001.SZ"))
    # print(st.etf_mins(ts_code="510300.SH"))
    # print(st.idx_mins(ts_code="000001.SH"))
    # print(st.opt_mins(ts_code="10007976.SH", freq="1min", start_date="2024-09-27 09:00:00", end_date="2024-09-27 15:00:00"))  # 期权分钟 241条
    # print(st.hk_adjfactor(ts_code="00700.HK"))
    # print(st.hk_fina_indicator(ts_code="00700.HK"))
    # print(st.us_adjfactor(ts_code="AAPL"))                            # 美股复权因子(需单独权限)
    # print(st.us_fina_indicator(ts_code="AAPL", period="20250331"))  # 美股财务指标 105条
    # print(st.us_tbr())
    # print(st.us_tycr())
    # print(st.us_trycr())
    # print(st.us_tltr())
    # print(st.us_trltr())
    # print(st.libor())
    # print(st.hibor())

    # ==================== 票房/电影数据 ====================
    # print(st.bo_daily(date="20260506"))                               # 电影日度票房
    # print(st.bo_weekly(date="20260506"))                              # 电影周度票房
    # print(st.bo_monthly(date="202605"))                               # 电影月度票房
    # print(st.bo_cinema(date="20260506"))                             # 影院日度票房
    # print(st.film_record(start_date="20260101", end_date="20260506"))  # 电影剧本备案
    # print(st.teleplay_record(start_date="20260101", end_date="20260506"))  # 电视剧备案

    # ==================== 东财/同花顺 ====================
    # print(st.dc_index(trade_date="20260506"))                         # 东财指数 486条(idx_type默认概念板块)
    # print(st.dc_member(ts_code="BK1184.DC", trade_date="20260506"))   # 东财成分股 50条

    # ==================== IRM/问答 ====================
    # print(st.irm_qa_sh(ts_code="000001.SZ"))                          # 上证e互动
    # print(st.irm_qa_sz(ts_code="000001.SZ"))                          # 深证e互动

    # ==================== 台湾数据 ====================
    # print(st.tmt_twincome(start_date="20260101", end_date="20260506"))  # 台湾电子月

    # ==================== 18. VIP财务数据 ====================
    # print(st.income_vip(ts_code="000001.SZ", period="20250331"))     # 利润表VIP
    # print(st.balancesheet_vip(ts_code="000001.SZ", period="20250331"))  # 资产负债表VIP
    # print(st.cashflow_vip(ts_code="000001.SZ", period="20250331"))   # 现金流量表VIP
    # print(st.fina_indicator_vip(ts_code="000001.SZ", period="20250331"))  # 财务指标VIP
    # print(st.express_vip(ts_code="000001.SZ"))       # 业绩快报VIP
    # print(st.forecast_vip(ts_code="000001.SZ"))       # 业绩预告VIP
    # print(st.fina_mainbz_vip(ts_code="000001.SZ"))    # 主营构成VIP

    print("\n=== 测试完成 ===")
