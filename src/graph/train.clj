(ns graph.train
  (:gen-class)
  (:require [clojure.data.json :as json]
            [graph.core]
            [clojure.pprint :as pp]))

(defrecord PersistentEdge [id mean sd original-mean from to])

(defrecord PersistentNode [id from to])

(defrecord PersistentGraph [nodes edges])

(defn make-edge [idx input]
  (->PersistentEdge idx (get input "mean") (get input "sd") (get input "mean") (get input "from") (get input "to")))

(defn make-node-from [edge nodes]
  (let [id (:from edge)]
    (if (contains? nodes id) (assoc-in nodes [id :to] (assoc (:to (get nodes id)) (:to edge) (:id edge))) (assoc nodes id (->PersistentNode id {} {(:to edge) (:id edge)})))))

(defn make-node-to [edge nodes]
  (let [id (:to edge)]
    (if (contains? nodes id) (assoc-in nodes [id :from] (assoc (:from (get nodes id)) (:from edge) (:id edge))) (assoc nodes id (->PersistentNode id {(:from edge) (:id edge)} {})))))

(defn make-nodes [edges nodes]
  (if (empty? edges) nodes (make-nodes (drop-last edges) (make-node-from (last edges) (make-node-to (last edges) nodes)))))

(defn parse-graph [parsed]
  (let [edges (map-indexed make-edge parsed)]
    (->PersistentGraph (make-nodes edges {}) (into {} (map (fn [edge] [(:id edge) edge]) edges)))))

(defn make-train-edges [edges]
  (map (fn [id] (let [edge (get edges id)] [(:mean edge) (:sd edge) (:original-mean edge) id])) (range (count edges))))

(defn make-train-graph [graph]
  (apply graph.core/add-edges (graph.core/add-nodes (graph.core/->Graph nil nil) (count (:nodes graph))) (make-train-edges (:edges graph))))

(defn make-terminal [node total]
  (graph.core/terminal-to (get node "id") (:in-degree node) total))

(defn edge-flow [node learning-rate cache]
  (fn [to]
    (graph.core/node-to (get (:to node) to) (get cache to) learning-rate)))

(defn node-flow [persistent-graph node learning-rate cache]
  (apply graph.core/edge-to (get node "id") (:in-degree node) (map (edge-flow (get (:nodes persistent-graph) (get node "id")) learning-rate cache) (get node "to"))))

(defn build-train-flow [persistent-graph nodes learning-rate cache]
  (if (empty? nodes) cache (build-train-flow persistent-graph (drop-last nodes) learning-rate (assoc cache (get (last nodes) "id") (node-flow persistent-graph (last nodes) learning-rate cache)))))

(defn make-train-subgraph [persistent-graph nodes learning-rate total]
  (get (build-train-flow persistent-graph (drop-last nodes) learning-rate {(get (last nodes) "id") (make-terminal (last nodes) total)}) (get (first nodes) "id")))

(defn call-subgraph [persistent-graph parsed learning-rate total]
  ((make-train-subgraph persistent-graph parsed learning-rate total) (graph.core/make-node-funcs (get (first parsed) "id") (make-train-graph persistent-graph)) 0))

(defn persist-edge [edges]
  (fn [[id edge]]
    [id (assoc (get edges id) :mean (:mean edge))]))

(defn persist-training [persistent-graph train-graph]
  (assoc persistent-graph :edges (into {} (map (persist-edge (:edges persistent-graph)) (:edges train-graph)))))

(defn node-in-degree [edges counts]
  (if (empty? edges) counts (node-in-degree (drop-last edges) (assoc counts (last edges) (+ (get counts (last edges) 0) 1)))))

(defn count-in-degree [subgraph counts]
  (if (empty? subgraph) counts (count-in-degree (drop-last subgraph) (node-in-degree (get (last subgraph) "to") counts))))

(defn add-in-degree [subgraph]
  (let [counts (count-in-degree subgraph {})]
    (map (fn [node] (assoc node :in-degree (get counts (get node "id") 1))) subgraph)))

(defn train-once [persistent-graph subgraph learning-rate total]
  (persist-training persistent-graph (apply graph.core/extract-graph (call-subgraph persistent-graph (add-in-degree subgraph) learning-rate total))))

(defn parse-example [parsed]
  [(get parsed "subgraph") (get parsed "learning-rate") (get parsed "total")])

(defn train-many [graph & examples]
  (if (empty? examples) graph (apply train-many (apply train-once graph (parse-example (last examples))) (drop-last examples))))

(defn train-from-json [input]
  (let [parsed (json/read-str input)]
    (apply train-many (parse-graph (get parsed "graph")) (get parsed "examples"))))

(defn -main [& args]
  (pp/pprint (train-from-json (slurp (first args)))))