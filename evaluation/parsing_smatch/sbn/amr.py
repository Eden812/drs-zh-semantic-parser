# -*- coding: utf-8 -*-

import penman


class AMR:
    def __init__(self, var_list, conc_list, relation_list, attribute_list):
        self.var_list = list(var_list)
        self.conc_list = list(conc_list)
        self.relation_list = relation_list
        self.attribute_list = attribute_list
        self.nodes = self.var_list[:]
        self.node_values = self.conc_list[:]

    def __str__(self):
        return " ".join(self.var_list)

    def rename_node(self, prefix):
        old_to_new = {}
        for i, old_var in enumerate(self.var_list):
            old_to_new[old_var] = f"{prefix}{i}"

        self.var_list = [old_to_new[v] for v in self.var_list]
        self.nodes = self.var_list[:]

        new_relation_list = []
        for rels in self.relation_list:
            cur = []
            for rel_name, tgt in rels:
                tgt_new = old_to_new[tgt] if tgt in old_to_new else tgt
                cur.append([rel_name, tgt_new])
            new_relation_list.append(cur)
        self.relation_list = new_relation_list

        new_attribute_list = []
        for attrs in self.attribute_list:
            cur = []
            for attr_name, val in attrs:
                cur.append([attr_name, val])
            new_attribute_list.append(cur)
        self.attribute_list = new_attribute_list

    def get_triples(self):
        instance_triples = []
        attribute_triples = []
        relation_triples = []

        for var, conc in zip(self.var_list, self.conc_list):
            instance_triples.append(("instance", var, conc))

        for var, attrs in zip(self.var_list, self.attribute_list):
            for attr_name, val in attrs:
                attribute_triples.append((attr_name, var, val))

        for var, rels in zip(self.var_list, self.relation_list):
            for rel_name, tgt in rels:
                relation_triples.append((rel_name, var, tgt))

        return instance_triples, attribute_triples, relation_triples

    @staticmethod
    def parse_AMR_line(penman_text):
        """
        兼容旧 smatch 代码的接口：
        输入一行 penman / amr 文本，返回 AMR 对象
        """
        penman_text = penman_text.replace("\n", " ").strip()
        if not penman_text:
            return None

        g = penman.decode(penman_text)

        var_list = []
        conc_list = []
        relation_dict = {}
        attribute_dict = {}

        # instance
        for src, role, tgt in g.instances():
            if src not in var_list:
                var_list.append(src)
                conc_list.append(tgt)
            relation_dict.setdefault(src, [])
            attribute_dict.setdefault(src, [])

        # relation edges
        for src, role, tgt in g.edges():
            role = role.lstrip(":")
            if src not in relation_dict:
                relation_dict[src] = []
            relation_dict[src].append([role, tgt])

            if tgt not in relation_dict:
                relation_dict.setdefault(tgt, [])
            attribute_dict.setdefault(src, [])
            attribute_dict.setdefault(tgt, [])

            if tgt not in var_list:
                # 理论上 edge 的 tgt 应该已在 instances 里
                var_list.append(tgt)
                conc_list.append(tgt)

        # attributes
        for src, role, tgt in g.attributes():
            role = role.lstrip(":")
            attribute_dict.setdefault(src, [])
            relation_dict.setdefault(src, [])
            attribute_dict[src].append([role, tgt])

        relation_list = [relation_dict.get(v, []) for v in var_list]
        attribute_list = [attribute_dict.get(v, []) for v in var_list]

        # 如果没有 TOP，就给第一个节点补一个
        if len(conc_list) > 0:
            has_top = any(attr_name == "TOP" for attrs in attribute_list for attr_name, _ in attrs)
            if not has_top:
                attribute_list[0].append(["TOP", conc_list[0]])

        return AMR(var_list, conc_list, relation_list, attribute_list)

    @staticmethod
    def get_amr_line(f):
        """
        兼容旧 smatch.py 的接口：
        - 既支持文件句柄（有 readline）
        - 也支持 list / iterator
        - 返回下一条非空字符串；没有了就返回 ""
        """
        # 情况1：文件对象
        if hasattr(f, "readline"):
            while True:
                line = f.readline()
                if not line:
                    return ""
                line = line.strip()
                if line:
                    return line

        # 情况2：list / tuple
        if isinstance(f, (list, tuple)):
            while len(f) > 0:
                line = f.pop(0)
                if line is None:
                    continue
                line = str(line).strip()
                if line:
                    return line
            return ""

        # 情况3：一般 iterator
        try:
            while True:
                line = next(f)
                if line is None:
                    continue
                line = str(line).strip()
                if line:
                    return line
        except StopIteration:
            return ""