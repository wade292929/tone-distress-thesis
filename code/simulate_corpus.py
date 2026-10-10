# -*- coding: utf-8 -*-
"""
模拟语料生成器 —— 企业年报「管理层讨论与分析」(MD&A) 文本

设计要点
--------
1. 每家公司(每年度)都有一个潜在的"经营困难程度" d ∈ (0,1)，d 越高，
   文本中消极词、不确定性词、风险合规词的投放强度越大，积极词的稀缺程度越高；
2. 文本篇幅(信息含量)由独立的对数字正态因子驱动，与 d 无关 —— 这样后续
   才可以做"区分效度"检验(测度应与信息含量这一无关构念不相关)；
3. 约 22% 的公司存在"语调操纵"：主动增加积极表述、压低消极表述，从而
   在"文本语调"与"基本面"之间制造系统性背离 —— 为后文的异常语调分解
   与操纵识别提供数据基础；
4. 全部随机数由固定种子(20261010)驱动，结果完全可复现。

输出
----
data/corpus/mda_2021.jsonl / mda_2022.jsonl / mda_2023.jsonl
        每行一条 JSON：{firm_id, firm_name, industry, year, text}
data/financials.csv     可观测的财务与治理变量(时间序列)
data/_latent_sim.csv    仿真潜在参数(真值)，仅供仿真说明使用，不进入分析样本
"""
import os
import csv
import json
import math
import random

SEED = 20261010
random.seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_PATH = os.path.join(ROOT, 'code', 'dictionary', 'tone_dict.txt')
CORPUS_DIR = os.path.join(ROOT, 'data', 'corpus')
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

N_FIRMS = 300
YEARS = [2021, 2022, 2023]
INDUSTRIES = ['制造业', '信息技术业', '批发零售业', '医药生物',
              '材料能源', '建筑业', '交通运输业', '农林牧渔业']
DISTRESS_RATE = 0.17          # 目标困境率
MANIP_SHARE = 0.22            # 语调操纵样本占比

PREFIX = ['华信', '中泰', '东旭', '南方', '北新', '恒瑞', '联合', '新和', '盛达', '汇通',
          '丰华', '宏图', '兴业', '正邦', '天泽', '海联', '金田', '龙腾', '瑞丰', '嘉禾',
          '同益', '三和', '长虹', '万邦', '众诚', '昌盛', '德润', '明远', '泰合', '晟源']
SUFFIX = ['科技', '实业', '电子', '材料', '医药', '能源', '机械', '食品', '化工', '智能']


# ---------------------------------------------------------------- 词典读取
def load_dict():
    cats, cur = {}, None
    with open(DICT_PATH, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#') and '=====' in s:
                cur = s.split('=====')[1].strip().split()[0]
                cats[cur] = []
            elif s.startswith('#'):
                continue
            elif cur:
                cats[cur].append(s)
    return cats


# ---------------------------------------------------------------- 句式模板
TPL = {
    'POS': [
        '报告期内，公司持续{0}核心主业，市场份额实现{1}，利润水平{2}。',
        '公司加大研发投入，{0}产品结构，技术壁垒不断{1}，竞争地位进一步{2}。',
        '面对外部环境变化，公司经营保持{0}，主要经营指标{1}，行业地位持续{2}。',
        '公司深入推进成本管控，运营效率实现{0}，客户结构{1}，发展动能{2}。',
        '报告期内，公司新签订单{0}，产能利用率{1}，规模效应逐步{2}。',
        '公司{0}渠道布局，品牌影响力{1}，市场份额{2}。',
        '报告期内，公司战略落地顺利，组织能力{0}，经营质量{1}，发展基础{2}。',
        '公司围绕主业{0}业务边界，产业链协同更加{1}，整体格局{2}。',
    ],
    'NEG': [
        '报告期内，公司经营{0}，营业收入出现{1}，毛利率水平{2}。',
        '受行业景气度回落影响，公司产品价格{0}，订单节奏{1}，经营{2}。',
        '报告期内，公司部分生产线{0}，库存周转{1}，营运效率{2}。',
        '受下游景气影响，公司销售收入{0}，应收账款{1}，现金流{2}。',
        '报告期内，公司对部分资产确认损失，利润总额{0}，净资产收益率{1}，股东回报{2}。',
        '公司所处行业竞争格局变化，市场份额{0}，议价能力{1}，经营{2}。',
        '报告期内，公司主要产品销量{0}，单位成本{1}，整体业绩{2}。',
    ],
    'UNC': [
        '上述事项对公司未来经营的影响{0}，后续进展{1}，投资者{2}。',
        '关于新产线投产进度，公司表示{0}完成调试，具体时间{1}，后续安排{2}。',
        '相关部门政策的落地效果{0}，对行业格局的影响{1}，公司{2}。',
        '该笔款项的回收情况{0}，公司{1}采取相应措施，最终结果{2}。',
        '报告期内，公司重大合同的执行进度{0}，剩余部分{1}，收入确认{2}。',
    ],
    'RISK': [
        '报告期内，公司与供应商之间存在{0}事项，涉案金额{1}，目前{2}。',
        '公司控股股东所持部分股份被{0}，占其持股比例{1}，公司{2}。',
        '报告期内，公司收到交易所{0}，公司已按要求{1}，相关事项{2}。',
        '公司对外{0}余额较大，其中{1}事项涉及{2}。',
        '报告期内，公司存在{0}情形，公司已开展{1}，并加强{2}。',
    ],
}

# 中性填充句：经人工核对，不含任何词典词项
FILLER = [
    '报告期内，公司股东大会、董事会及各专门委员会规范运作，治理结构进一步完善。',
    '公司严格按照相关规定履行信息披露义务，内部控制体系保持稳定运行。',
    '报告期内，公司完成董事会换届选举，管理团队保持稳定。',
    '公司积极履行社会责任，在环境保护与员工关怀方面开展多项工作。',
    '报告期内，公司组织多场员工培训与团队建设活动。',
    '公司办公场所与主要生产基地位于华东地区，员工总数与上年基本持平。',
    '报告期内，公司未发生重大安全与环保事故。',
    '公司财务报表按照企业会计准则编制，并由年审会计师事务所出具审计意见。',
    '报告期内，公司主要客户结构与上年相比保持稳定。',
    '公司持续健全信息化管理系统，日常运营流程更加规范。',
    '报告期内，公司参加多场行业交流活动，与同行的沟通进一步加强。',
    '公司设有独立董事三名，报告期内均亲自出席董事会会议。',
    '公司注册地址位于省会城市，报告期内未发生变更。',
    '报告期内，公司完成年度预算编制与执行情况评估。',
    '公司按照既定计划推进园区基础设施维护工作。',
    '报告期内，公司员工薪酬体系保持稳定，绩效考核按季度开展。',
]


def poisson(lam):
    """Knuth 算法：从 Poisson(lam) 抽样（纯标准库实现）"""
    lam = max(0.0, float(lam))
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1
        if k > 400:
            return k


def make_sentence(cat, words):
    t = random.choice(TPL[cat])
    n = t.count('{')
    picks = []
    for _ in range(n):
        picks.append(random.choice(words))
    return t.format(*picks)


def make_text(name, year, d, length_factor, manip, dicts):
    """按潜在困难程度 d 生成一篇 MD&A 文本"""
    pos_boost = 1.55 if manip else 1.0
    neg_damp = 0.58 if manip else 1.0

    n_pos = poisson(max(0.6, (2.4 + 15.0 * (1.0 - d)) * pos_boost * length_factor))
    n_neg = poisson(max(0.3, (1.9 + 12.5 * d) * neg_damp * length_factor))
    n_unc = poisson(max(0.2, (1.2 + 7.0 * d) * length_factor))
    n_risk = poisson(max(0.1, (0.4 + 4.6 * d) * length_factor))
    n_fill = poisson(9.0 * length_factor)

    body = []
    for _ in range(n_pos):
        body.append(make_sentence('POS', dicts['POS']))
    for _ in range(n_neg):
        body.append(make_sentence('NEG', dicts['NEG']))
    for _ in range(n_unc):
        body.append(make_sentence('UNC', dicts['UNC']))
    for _ in range(n_risk):
        body.append(make_sentence('RISK', dicts['RISK']))
    for _ in range(n_fill):
        body.append(random.choice(FILLER))
    random.shuffle(body)

    head = ('%s（%s）%s 年年度报告 第三节 管理层讨论与分析\n\n'
            '一、经营情况讨论与分析\n') % (name, '证券代码', year)
    para = []
    for i in range(0, len(body), 3):
        para.append(''.join(body[i:i + 3]))
    return head + '\n'.join(para) + '\n'


def main():
    dicts = load_dict()
    print('[词典] ' + ' · '.join('%s %d 词' % (k, len(v)) for k, v in sorted(dicts.items())))
    print('[词典] 合计 %d 个词项' % sum(len(v) for v in dicts.values()))

    firms = []
    used = set()
    i = 0
    while len(firms) < N_FIRMS:
        nm = random.choice(PREFIX) + random.choice(SUFFIX)
        if nm in used:
            continue
        used.add(nm)
        firms.append({'firm_id': '%06d' % (600001 + i),
                      'firm_name': nm,
                      'industry': random.choice(INDUSTRIES),
                      'quality': random.gauss(0, 1.0),      # 潜在经营质量
                      'hidden': random.gauss(0, 1.0),       # 隐藏风险因子
                      'manip': 1 if random.random() < MANIP_SHARE else 0})
        i += 1

    for f in firms:
        c = random.random()
        f['age'] = random.randint(3, 8) if c < 0.30 else (random.randint(9, 15) if c < 0.75 else random.randint(16, 26))
        f['soe'] = 1 if random.random() < 0.34 else 0
        f['top1'] = min(0.72, max(0.08, random.gauss(0.34, 0.13) + 0.05 * f['soe']))
        f['dual'] = 1 if random.random() < 0.26 else 0

    L_year = {2021: -0.18, 2022: 0.10, 2023: 0.34}
    ROA_year = {2021: 0.006, 2022: -0.004, 2023: -0.012}

    rows = []       # 潜在真值
    fin = []        # 可观测变量
    L_all = []

    for f in firms:
        for y in YEARS:
            L = -0.95 * f['quality'] + 1.15 * f['hidden'] + L_year[y] + random.gauss(0, 0.50)
            # 困境标签由"文本可反映的风险"与"文本无法反映的私有风险"共同决定，
            # 使文本语调只是困境的一个有噪信号（贴近真实情形，避免 AUC 虚高）
            L_dist = 0.80 * L + 0.70 * random.gauss(0, 1.0)
            d = 1.0 / (1.0 + math.exp(-L))
            lf = min(2.2, max(0.55, math.exp(random.gauss(0, 0.28))))
            L_all.append(L_dist)
            rows.append({'firm_id': f['firm_id'], 'year': y, 'L_text': L, 'L_distress': L_dist,
                         'd': d, 'quality': f['quality'], 'hidden': f['hidden'],
                         'manip': f['manip'], 'length_factor': lf})

            roa = 0.062 - 0.021 * f['quality'] - 0.011 * f['hidden'] + ROA_year[y] + random.gauss(0, 0.018)
            lev = 0.44 - 0.055 * f['quality'] + 0.045 * f['hidden'] + random.gauss(0, 0.075)
            lev = min(0.92, max(0.05, lev))
            growth = 0.11 + 0.075 * f['quality'] - 0.045 * f['hidden'] + random.gauss(0, 0.13)
            size = 13.0 + 0.42 * f['quality'] + random.gauss(0, 0.75)
            fin.append({'firm_id': f['firm_id'], 'firm_name': f['firm_name'],
                        'industry': f['industry'], 'year': y, 'age': f['age'],
                        'soe': f['soe'], 'dual': f['dual'],
                        'top1': round(f['top1'], 4), 'size': round(size, 4),
                        'assets': round(math.exp(size), 2),
                        'roa': round(roa, 5), 'lev': round(lev, 4),
                        'growth': round(growth, 4)})

    # 困境标签：按潜在风险 L 的 83% 分位切分（次年是否被实施特别处理/陷入困境）
    L_sorted = sorted(L_all)
    cut = L_sorted[int(len(L_sorted) * (1 - DISTRESS_RATE))]
    flag = {id(r): (1 if r['L_distress'] >= cut else 0) for r in rows}

    # 写语料
    for y in YEARS:
        path = os.path.join(CORPUS_DIR, 'mda_%d.jsonl' % y)
        with open(path, 'w', encoding='utf-8') as fo:
            for f in firms:
                r = [x for x in rows if x['firm_id'] == f['firm_id'] and x['year'] == y][0]
                txt = make_text(f['firm_name'], y, r['d'], r['length_factor'], f['manip'], dicts)
                fo.write(json.dumps({'firm_id': f['firm_id'], 'firm_name': f['firm_name'],
                                     'industry': f['industry'], 'year': y, 'text': txt},
                                    ensure_ascii=False) + '\n')
        print('[语料] 已写出 %s' % os.path.basename(path))

    # 写可观测财务变量
    with open(os.path.join(DATA_DIR, 'financials.csv'), 'w', encoding='utf-8-sig', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(fin[0].keys()))
        w.writeheader()
        w.writerows(fin)

    # 合并困境标签后写出（标签属于"结果变量"，与文本本身分离存放）
    with open(os.path.join(DATA_DIR, '_latent_sim.csv'), 'w', encoding='utf-8-sig', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_dist = sum(flag[id(r)] for r in rows)
    print('[财务] financials.csv  %d 行' % len(fin))
    print('[真值] _latent_sim.csv %d 行' % len(rows))
    print('[标记] 困境样本 %d / %d = %.2f%%' % (n_dist, len(rows), 100.0 * n_dist / len(rows)))

    # 困境标签单列一份（供流水线 S5 使用）
    with open(os.path.join(DATA_DIR, 'distress_label.csv'), 'w', encoding='utf-8-sig', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['firm_id', 'year', 'distress_next'])
        for r in rows:
            w.writerow([r['firm_id'], r['year'], flag[id(r)]])
    print('[标签] distress_label.csv 已写出')


if __name__ == '__main__':
    main()
