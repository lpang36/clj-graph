(ns graph.core
  (:gen-class))

(require '[incanter.stats :as stats])
(require '[clojure.pprint :as pp])

(defrecord EdgeState [id mean sd original-mean back])

(defrecord NodeState [id count received-count max back])

(defrecord Graph [nodes edges])

(defn sample [getter]
  (stats/sample-normal 1 :mean (getter :mean) :sd (getter :sd)))

(defn forward-edge [from getter]
  (+ from (sample getter)))

(defn abs [n]
  (if (> 0 n) n (- n)))

(defn dist [getter val]
  (abs (/ (- val (getter :mean)) (getter :sd))))

(defn linterp [frac from to]
  (+ from (* frac (- to from))))

(defn adj-mean [getter from to]
  (if (> (dist getter from) (dist getter to)) to (linterp (/ (dist getter from) (dist getter to)) from to)))

(defn backward-edge [learning-rate]
  (fn [[getter setter] loss]
    [(setter :mean (adj-mean getter (getter :mean) (- (getter :mean) (* learning-rate loss)))) loss]))

(defn maybe-learn [edge-id learning-rate back]
  (fn [[getter setter] loss]
    (if (= loss 0) [[getter setter] 0] (apply back ((backward-edge learning-rate) (getter :edge-for edge-id) loss)))))

(defn run-edge [[getter setter] from to learning-rate back]
  (to (setter :back (maybe-learn (getter :id) learning-rate back)) (forward-edge from getter)))

(defn update-node [[getter setter] from back]
  (if (> from (getter :max)) (setter :max from :back back :received-count (+ 1 (getter :received-count))) (setter :received-count (+ 1 (getter :received-count)))))

(defn unready-node [[getter setter] & to]
  [[getter setter] 0])

(defn multi-to [[getter setter] val node-id to]
  (if (empty? to) [(getter :node-for node-id) val] (apply multi-to (concat ((last to) (getter :node-for node-id) val) [node-id (drop-last to)]))))

(defn ready-node [[getter setter] & to]
  (multi-to [getter setter] (getter :max) (getter :id) to))

(defn run-node-helper [[getter setter] & to]
  (apply (if (= (getter :received-count) (getter :count)) ready-node unready-node) [getter setter] to))

(defn run-node [[getter setter] from back & to]
  (apply run-node-helper (update-node [getter setter] from back) to))

(defn eval-terminal [total]
  (fn [[getter setter] val]
    ((getter :back) [getter setter] (- val total))))

(defn terminal-to [node-id in-degree total]
  (fn [[getter setter] val]
    (run-node ((last (getter :node-for node-id)) :count in-degree) val (getter :back) (eval-terminal total))))

(defn edge-to [node-id in-degree & to]
  (fn [[getter setter] val]
    (apply run-node ((last (getter :node-for node-id)) :count in-degree) val (getter :back) to)))

(defn node-to [edge-id to learning-rate]
  (fn [[getter setter] val]
    (run-edge (getter :edge-for edge-id) val to learning-rate (getter :back))))

(declare make-node-funcs make-edge-funcs make-edge-setter make-node-setter)

(defn make-generic-getter [attr id graph]
  (if (= attr :node-for) (make-node-funcs id graph) (if (= attr :edge-for) (make-edge-funcs id graph) graph)))

(defn make-edge-getter [edge-id graph]
  (fn
    ([attr] (attr (get (:edges graph) edge-id)))
    ([attr id] (make-generic-getter attr id graph))))

(defn make-node-getter [node-id graph]
  (fn
    ([attr] (attr (get (:nodes graph) node-id)))
    ([attr id] (make-generic-getter attr id graph))))

(defn make-edge-funcs [id graph]
  [(make-edge-getter id graph) (make-edge-setter id graph)])

(defn make-node-funcs [id graph]
  [(make-node-getter id graph) (make-node-setter id graph)])

(defn make-edge-setter [edge-id graph]
  (fn [& args]
    (make-edge-funcs edge-id (assoc-in graph [:edges edge-id] (apply assoc (get (:edges graph) edge-id) args)))))

(defn make-node-setter [node-id graph]
  (fn [& args]
    (make-node-funcs node-id (assoc-in graph [:nodes node-id] (apply assoc (get (:nodes graph) node-id) args)))))

(defn multi-identity [& args]
  args)

(defn add-node
  ([graph id] (assoc-in graph [:nodes id] (->NodeState id 0 0 0 multi-identity)))
  ([graph] (add-node graph (count (:nodes graph)))))

(defn add-nodes [graph n]
  (if (> n 0) (add-nodes (add-node graph) (- n 1)) graph))

(defn add-edge
  ([graph mean sd id] (assoc-in graph [:edges id] (->EdgeState id mean sd mean nil)))
  ([graph mean sd] (add-edge graph mean sd (count (:edges graph)))))

(defn add-edges [graph & args]
  (if (empty? args) graph (apply add-edge (apply add-edges graph (rest args)) (first args))))

(defn test-graph []
  (add-edges (add-nodes (->Graph nil nil) 4) [1 0.5] [2 0.1] [1.5 1] [1.5 0.2]))

(defn test-terminal []
  (terminal-to 3 2 3))

(defn test-flow [graph learning-rate terminal]
  ((edge-to 0 1 (node-to 0 (edge-to 1 1 (node-to 2 terminal learning-rate)) learning-rate) (node-to 1 (edge-to 2 1 (node-to 3 terminal learning-rate)) learning-rate)) (make-node-funcs 0 graph) 0))

(defn extract-graph [[getter setter] loss]
  (getter :graph 0))

(defn -main [& args]
  (pp/pprint (apply extract-graph (test-flow (test-graph) 0.1 (test-terminal)))))