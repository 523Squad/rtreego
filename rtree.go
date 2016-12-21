// Copyright 2012 Daniel Connelly.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A library for efficiently storing and querying spatial data.
package rtreego

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
)

// Rtree represents an R-tree, a balanced search tree for storing and querying
// spatial objects.  Dim specifies the number of spatial dimensions and
// MinChildren/MaxChildren specify the minimum/maximum branching factors.
type Rtree struct {
	Dim         int
	MinChildren int
	MaxChildren int
	Root        *node
	TreeSize    int
	Height      int
}

// NewTree creates a new R-tree instance.
func NewTree(Dim, MinChildren, MaxChildren int) *Rtree {
	rt := Rtree{Dim: Dim, MinChildren: MinChildren, MaxChildren: MaxChildren}
	rt.Height = 1
	rt.Root = &node{}
	rt.Root.Entries = []entry{}
	rt.Root.Leaf = true
	rt.Root.Level = 1
	return &rt
}

// Size returns the number of objects currently stored in tree.
func (tree *Rtree) Size() int {
	return tree.TreeSize
}

func (tree *Rtree) String() string {
	return fmt.Sprintf(`{"Dim":%d,"MinChildren":%d,"MaxChildren":%d,"Size":%d,"Height":%d,"Root":%v}`,
		tree.Dim, tree.MinChildren, tree.MaxChildren, tree.TreeSize, tree.Height, tree.Root)
}

// Depth returns the maximum depth of tree.
func (tree *Rtree) Depth() int {
	return tree.Height
}

func (tree *Rtree) Equals(cmp Comparable) (bool, error) {
	that, ok := cmp.(*Rtree)
	if !ok {
		return false, fmt.Errorf("cmp is not *Rtree")
	}
	if that == nil {
		return false, fmt.Errorf("that is nil")
	}
	if tree == that {
		return true, nil
	}
	res := tree.Dim == that.Dim &&
		tree.MinChildren == that.MinChildren &&
		tree.MaxChildren == that.MaxChildren &&
		tree.TreeSize == that.TreeSize &&
		tree.Height == that.Height
	if !res {
		return false, fmt.Errorf(fmt.Sprintf("tree's and that's primitive fields are not equal: %+v != %+v", tree, that))
	}
	return tree.Root.Equals(that.Root)
}

// node represents a tree node of an Rtree.
type node struct {
	Parent  *node `json:"-"`
	Leaf    bool
	Entries []entry
	Level   int // node depth in the Rtree
}

func (n *node) UnmarshalJSON(b []byte) error {
	var data map[string]*json.RawMessage
	json.Unmarshal(b, &data)
	json.Unmarshal(*data["Leaf"], &n.Leaf)
	json.Unmarshal(*data["Level"], &n.Level)
	var es []*entry
	json.Unmarshal(*data["Entries"], &es)
	for _, e := range es {
		if e.Child != nil {
			e.Child.Parent = n
		}
		n.Entries = append(n.Entries, *e)
	}
	return nil
}

func (n *node) String() string {
	return fmt.Sprintf("{\"Node\": {\"Leaf\": %v,\n\t\"Entries\": %v\n}}\n", n.Leaf, n.Entries)
}

func (n *node) Equals(cmp Comparable) (bool, error) {
	that, ok := cmp.(*node)
	if !ok {
		return false, fmt.Errorf("cmp is not *node")
	}
	if that == nil {
		return false, fmt.Errorf("that is nil")
	}
	if n == that {
		return true, nil
	}
	res := n.Leaf == that.Leaf && n.Level == that.Level &&
		// Actual check needs to be cached if performed fairly, so believe iteration
		// order for now and that both Parents of `n` and `that` point either to nil
		// or to semantically the same node.
		((n.Parent == nil) == (that.Parent == nil))
	var err error
	if res {
		for i, e := range n.Entries {
			res = res && i < len(that.Entries)
			if !res || err != nil {
				break
			}
			res, err = e.Equals(that.Entries[i])
		}
	} else {
		err = fmt.Errorf("primitives are not equal")
	}
	return res, err
}

// entry represents a spatial index record stored in a tree node.
type entry struct {
	BoundingBox *Rect   // bounding-box of all children of this entry
	Child       *node   `json:",omitempty"`
	Obj         Spatial `json:",omitempty"`
}

func (e *entry) UnmarshalJSON(b []byte) error {
	var data map[string]*json.RawMessage
	err := json.Unmarshal(b, &data)
	if err != nil {
		return err
	}
	err = json.Unmarshal(*data["BoundingBox"], &e.BoundingBox)
	if err != nil {
		return err
	}
	if s, ok := data["Obj"]; ok {
		e.Obj = &SPoint{}
		if err := json.Unmarshal(*s, &e.Obj); err != nil {
			return err
		}
	}
	if c, ok := data["Child"]; ok {
		if err := json.Unmarshal(*c, &e.Child); err != nil {
			return err
		}
	}
	return nil
}

func (e entry) String() string {
	if e.Child != nil {
		return fmt.Sprintf("\"Entry\":\n\t{\"BoundingBox\": %v,\n\t\"Child\": %v\n}\n", e.BoundingBox, e.Child)
	}
	return fmt.Sprintf("\"Entry\":\n\t{\"BoundingBox\": %v,\n\t\"Obj\": %v\n}\n", e.BoundingBox, e.Obj)
}

func (e entry) Equals(cmp Comparable) (bool, error) {
	that, ok := cmp.(entry)
	if !ok {
		return false, fmt.Errorf("cmp is not entry")
	}
	if res, err := e.BoundingBox.Equals(that.BoundingBox); !res {
		return res, err
	}

	fmt.Println(e.Obj, that.Obj)
	if e.Child != nil && that.Child != nil {
		if res, err := e.Child.Equals(that.Child); !res {
			return res, err
		}
	} else if res, err := e.Obj.Equals(that.Obj); !res {
		return res, err
	}
	return true, nil
}

// Comparable allows to deep compare implementations.
type Comparable interface {
	Equals(Comparable) (bool, error)
}

// Any type that implements Spatial can be stored in an Rtree and queried.
type Spatial interface {
	Comparable
	Bounds() *Rect
}

// SPoint represents geographical point on the globe.
type SPoint struct {
	Latitude  float64
	Longitude float64
	Name      string `json:",omitempty"`
}

const sPointRadius = 0.001

// Bounds implement Spatial interface for *SPoint.
func (p *SPoint) Bounds() *Rect {
	// WTF, I need points, gives me error when widths are 0.
	var err error
	var bounds *Rect
	if bounds, err = NewRect(
		Point{p.Latitude - sPointRadius, p.Longitude - sPointRadius},
		[]float64{2 * sPointRadius, 2 * sPointRadius}); err == nil {
		return bounds
	}
	panic(err)
}

// Equals implement Spatial interface for *SPoint.
func (p *SPoint) Equals(cmp Comparable) (bool, error) {
	if that, ok := cmp.(*SPoint); ok {
		return p.Latitude == that.Latitude && p.Longitude == that.Longitude && p.Name == that.Name, nil
	}
	return false, fmt.Errorf("Wrong type of cmp: trying to compare *SPoint and %T", cmp)
}

func (p *SPoint) String() string {
	return fmt.Sprintf("%s, % .2f, % .2f", p.Name, p.Latitude, p.Longitude)
}

// Insertion

// Insert inserts a spatial object into the tree.  If insertion
// causes a leaf node to overflow, the tree is rebalanced automatically.
//
// Implemented per Section 3.2 of "R-trees: A Dynamic Index Structure for
// Spatial Searching" by A. Guttman, Proceedings of ACM SIGMOD, p. 47-57, 1984.
func (tree *Rtree) Insert(obj Spatial) {
	e := entry{obj.Bounds(), nil, obj}
	tree.insert(e, 1)
	tree.TreeSize++
}

// insert adds the specified entry to the tree at the specified level.
func (tree *Rtree) insert(e entry, level int) {
	leaf := tree.chooseNode(tree.Root, e, level)
	leaf.Entries = append(leaf.Entries, e)

	// update parent pointer if necessary
	if e.Child != nil {
		e.Child.Parent = leaf
	}

	// split leaf if overflows
	var split *node
	if len(leaf.Entries) > tree.MaxChildren {
		leaf, split = leaf.split(tree.MinChildren)
	}
	root, splitRoot := tree.adjustTree(leaf, split)
	if splitRoot != nil {
		oldRoot := root
		tree.Height++
		tree.Root = &node{
			Parent: nil,
			Level:  tree.Height,
			Entries: []entry{
				entry{BoundingBox: oldRoot.computeBoundingBox(), Child: oldRoot},
				entry{BoundingBox: splitRoot.computeBoundingBox(), Child: splitRoot},
			},
		}
		oldRoot.Parent = tree.Root
		splitRoot.Parent = tree.Root
	}
}

// chooseNode finds the node at the specified level to which e should be added.
func (tree *Rtree) chooseNode(n *node, e entry, level int) *node {
	if n.Leaf || n.Level == level {
		return n
	}

	// find the entry whose bb needs least enlargement to include obj
	diff := math.MaxFloat64
	var chosen entry
	for _, en := range n.Entries {
		bb := boundingBox(en.BoundingBox, e.BoundingBox)
		d := bb.size() - en.BoundingBox.size()
		if d < diff || (d == diff && en.BoundingBox.size() < chosen.BoundingBox.size()) {
			diff = d
			chosen = en
		}
	}

	return tree.chooseNode(chosen.Child, e, level)
}

// adjustTree splits overflowing nodes and propagates the changes upwards.
func (tree *Rtree) adjustTree(n, nn *node) (*node, *node) {
	// Let the caller handle root adjustments.
	if n == tree.Root {
		return n, nn
	}

	// Re-size the bounding box of n to account for lower-level changes.
	en := n.getEntry()
	en.BoundingBox = n.computeBoundingBox()

	// If nn is nil, then we're just propagating changes upwards.
	if nn == nil {
		return tree.adjustTree(n.Parent, nil)
	}

	// Otherwise, these are two nodes resulting from a split.
	// n was reused as the "left" node, but we need to add nn to n.parent.
	enn := entry{nn.computeBoundingBox(), nn, nil}
	n.Parent.Entries = append(n.Parent.Entries, enn)

	// If the new entry overflows the parent, split the parent and propagate.
	if len(n.Parent.Entries) > tree.MaxChildren {
		return tree.adjustTree(n.Parent.split(tree.MinChildren))
	}

	// Otherwise keep propagating changes upwards.
	return tree.adjustTree(n.Parent, nil)
}

// getEntry returns a pointer to the entry for the node n from n's parent.
func (n *node) getEntry() *entry {
	var e *entry
	for i := range n.Parent.Entries {
		if n.Parent.Entries[i].Child == n {
			e = &n.Parent.Entries[i]
			break
		}
	}
	return e
}

// computeBoundingBox finds the MBR of the children of n.
func (n *node) computeBoundingBox() (bb *Rect) {
	childBoxes := make([]*Rect, len(n.Entries))
	for i, e := range n.Entries {
		childBoxes[i] = e.BoundingBox
	}
	bb = boundingBoxN(childBoxes...)
	return
}

// split splits a node into two groups while attempting to minimize the
// bounding-box area of the resulting groups.
func (n *node) split(minGroupSize int) (left, right *node) {
	// find the initial split
	l, r := n.pickSeeds()
	leftSeed, rightSeed := n.Entries[l], n.Entries[r]

	// get the entries to be divided between left and right
	remaining := append(n.Entries[:l], n.Entries[l+1:r]...)
	remaining = append(remaining, n.Entries[r+1:]...)

	// setup the new split nodes, but re-use n as the left node
	left = n
	left.Entries = []entry{leftSeed}
	right = &node{
		Parent:  n.Parent,
		Leaf:    n.Leaf,
		Level:   n.Level,
		Entries: []entry{rightSeed},
	}

	// TODO
	if rightSeed.Child != nil {
		rightSeed.Child.Parent = right
	}
	if leftSeed.Child != nil {
		leftSeed.Child.Parent = left
	}

	// distribute all of n's old entries into left and right.
	for len(remaining) > 0 {
		next := pickNext(left, right, remaining)
		e := remaining[next]

		if len(remaining)+len(left.Entries) <= minGroupSize {
			assign(e, left)
		} else if len(remaining)+len(right.Entries) <= minGroupSize {
			assign(e, right)
		} else {
			assignGroup(e, left, right)
		}

		remaining = append(remaining[:next], remaining[next+1:]...)
	}

	return
}

func assign(e entry, group *node) {
	if e.Child != nil {
		e.Child.Parent = group
	}
	group.Entries = append(group.Entries, e)
}

// assignGroup chooses one of two groups to which a node should be added.
func assignGroup(e entry, left, right *node) {
	leftBB := left.computeBoundingBox()
	rightBB := right.computeBoundingBox()
	leftEnlarged := boundingBox(leftBB, e.BoundingBox)
	rightEnlarged := boundingBox(rightBB, e.BoundingBox)

	// first, choose the group that needs the least enlargement
	leftDiff := leftEnlarged.size() - leftBB.size()
	rightDiff := rightEnlarged.size() - rightBB.size()
	if diff := leftDiff - rightDiff; diff < 0 {
		assign(e, left)
		return
	} else if diff > 0 {
		assign(e, right)
		return
	}

	// next, choose the group that has smaller area
	if diff := leftBB.size() - rightBB.size(); diff < 0 {
		assign(e, left)
		return
	} else if diff > 0 {
		assign(e, right)
		return
	}

	// next, choose the group with fewer entries
	if diff := len(left.Entries) - len(right.Entries); diff <= 0 {
		assign(e, left)
		return
	}
	assign(e, right)
}

// pickSeeds chooses two child entries of n to start a split.
func (n *node) pickSeeds() (int, int) {
	left, right := 0, 1
	maxWastedSpace := -1.0
	for i, e1 := range n.Entries {
		for j, e2 := range n.Entries[i+1:] {
			d := boundingBox(e1.BoundingBox, e2.BoundingBox).size() - e1.BoundingBox.size() - e2.BoundingBox.size()
			if d > maxWastedSpace {
				maxWastedSpace = d
				left, right = i, j+i+1
			}
		}
	}
	return left, right
}

// pickNext chooses an entry to be added to an entry group.
func pickNext(left, right *node, entries []entry) (next int) {
	maxDiff := -1.0
	leftBB := left.computeBoundingBox()
	rightBB := right.computeBoundingBox()
	for i, e := range entries {
		d1 := boundingBox(leftBB, e.BoundingBox).size() - leftBB.size()
		d2 := boundingBox(rightBB, e.BoundingBox).size() - rightBB.size()
		d := math.Abs(d1 - d2)
		if d > maxDiff {
			maxDiff = d
			next = i
		}
	}
	return
}

// Deletion

// Delete removes an object from the tree.  If the object is not found, returns
// false, otherwise returns true.
//
// Implemented per Section 3.3 of "R-trees: A Dynamic Index Structure for
// Spatial Searching" by A. Guttman, Proceedings of ACM SIGMOD, p. 47-57, 1984.
func (tree *Rtree) Delete(obj Spatial) bool {
	n := tree.findLeaf(tree.Root, obj)
	if n == nil {
		return false
	}

	ind := -1
	for i, e := range n.Entries {
		if e.Obj == obj {
			ind = i
		}
	}
	if ind < 0 {
		return false
	}

	n.Entries = append(n.Entries[:ind], n.Entries[ind+1:]...)

	tree.condenseTree(n)
	tree.TreeSize--

	if !tree.Root.Leaf && len(tree.Root.Entries) == 1 {
		tree.Root = tree.Root.Entries[0].Child
	}

	return true
}

// findLeaf finds the leaf node containing obj.
func (tree *Rtree) findLeaf(n *node, obj Spatial) *node {
	if n.Leaf {
		return n
	}
	// if not leaf, search all candidate subtrees
	for _, e := range n.Entries {
		if e.BoundingBox.containsRect(obj.Bounds()) {
			leaf := tree.findLeaf(e.Child, obj)
			if leaf == nil {
				continue
			}
			// check if the leaf actually contains the object
			for _, leafEntry := range leaf.Entries {
				if leafEntry.Obj == obj {
					return leaf
				}
			}
		}
	}
	return nil
}

// condenseTree deletes underflowing nodes and propagates the changes upwards.
func (tree *Rtree) condenseTree(n *node) {
	deleted := []*node{}

	for n != tree.Root {
		if len(n.Entries) < tree.MinChildren {
			// remove n from parent entries
			entries := []entry{}
			for _, e := range n.Parent.Entries {
				if e.Child != n {
					entries = append(entries, e)
				}
			}
			if len(n.Parent.Entries) == len(entries) {
				panic(fmt.Errorf("Failed to remove entry from parent"))
			}
			n.Parent.Entries = entries

			// only add n to deleted if it still has children
			if len(n.Entries) > 0 {
				deleted = append(deleted, n)
			}
		} else {
			// just a child entry deletion, no underflow
			n.getEntry().BoundingBox = n.computeBoundingBox()
		}
		n = n.Parent
	}

	for _, n := range deleted {
		// reinsert entry so that it will remain at the same level as before
		e := entry{n.computeBoundingBox(), n, nil}
		tree.insert(e, n.Level+1)
	}
}

// Searching

// SearchIntersectBB returns all objects that intersect the specified rectangle.
//
// Implemented per Section 3.1 of "R-trees: A Dynamic Index Structure for
// Spatial Searching" by A. Guttman, Proceedings of ACM SIGMOD, p. 47-57, 1984.
func (tree *Rtree) SearchIntersect(bb *Rect) []Spatial {
	return tree.searchIntersect(-1, tree.Root, bb)
}

// SearchIntersectWithLimit is similar to SearchIntersect, but returns
// immediately when the first k results are found. A negative k behaves exactly
// like SearchIntersect and returns all the results.
func (tree *Rtree) SearchIntersectWithLimit(k int, bb *Rect) []Spatial {
	return tree.searchIntersect(k, tree.Root, bb)
}

func (tree *Rtree) searchIntersect(k int, n *node, bb *Rect) []Spatial {
	results := []Spatial{}
	for _, e := range n.Entries {
		if k >= 0 && len(results) >= k {
			break
		}

		if intersect(e.BoundingBox, bb) != nil {
			if n.Leaf {
				results = append(results, e.Obj)
			} else {
				margin := k - len(results)
				results = append(results, tree.searchIntersect(margin, e.Child, bb)...)
			}
		}
	}
	return results
}

// NearestNeighbor returns the closest object to the specified point.
// Implemented per "Nearest Neighbor Queries" by Roussopoulos et al
func (tree *Rtree) NearestNeighbor(p Point) Spatial {
	obj, _ := tree.nearestNeighbor(p, tree.Root, math.MaxFloat64, nil)
	return obj
}

// utilities for sorting slices of entries

type entrySlice struct {
	entries []entry
	dists   []float64
	pt      Point
}

func (s entrySlice) Len() int { return len(s.entries) }

func (s entrySlice) Swap(i, j int) {
	s.entries[i], s.entries[j] = s.entries[j], s.entries[i]
	s.dists[i], s.dists[j] = s.dists[j], s.dists[i]
}

func (s entrySlice) Less(i, j int) bool {
	return s.dists[i] < s.dists[j]
}

func sortEntries(p Point, entries []entry) ([]entry, []float64) {
	sorted := make([]entry, len(entries))
	dists := make([]float64, len(entries))
	for i := 0; i < len(entries); i++ {
		sorted[i] = entries[i]
		dists[i] = p.minDist(entries[i].BoundingBox)
	}
	sort.Sort(entrySlice{sorted, dists, p})
	return sorted, dists
}

func pruneEntries(p Point, entries []entry, minDists []float64) []entry {
	minMinMaxDist := math.MaxFloat64
	for i := range entries {
		minMaxDist := p.minMaxDist(entries[i].BoundingBox)
		if minMaxDist < minMinMaxDist {
			minMinMaxDist = minMaxDist
		}
	}
	// remove all entries with minDist > minMinMaxDist
	pruned := []entry{}
	for i := range entries {
		if minDists[i] <= minMinMaxDist {
			pruned = append(pruned, entries[i])
		}
	}
	return pruned
}

func (tree *Rtree) nearestNeighbor(p Point, n *node, d float64, nearest Spatial) (Spatial, float64) {
	if n.Leaf {
		for _, e := range n.Entries {
			dist := math.Sqrt(p.minDist(e.BoundingBox))
			if dist < d {
				d = dist
				nearest = e.Obj
			}
		}
	} else {
		branches, dists := sortEntries(p, n.Entries)
		branches = pruneEntries(p, branches, dists)
		for _, e := range branches {
			subNearest, dist := tree.nearestNeighbor(p, e.Child, d, nearest)
			if dist < d {
				d = dist
				nearest = subNearest
			}
		}
	}

	return nearest, d
}

func (tree *Rtree) NearestNeighbors(k int, p Point) []Spatial {
	dists := make([]float64, k)
	objs := make([]Spatial, k)
	for i := 0; i < k; i++ {
		dists[i] = math.MaxFloat64
		objs[i] = nil
	}
	objs, _ = tree.nearestNeighbors(k, p, tree.Root, dists, objs)
	return objs
}

// insert obj into nearest and return the first k elements in increasing order.
func insertNearest(k int, dists []float64, nearest []Spatial, dist float64, obj Spatial) ([]float64, []Spatial) {
	i := 0
	for i < k && dist >= dists[i] {
		i++
	}
	if i >= k {
		return dists, nearest
	}

	left, right := dists[:i], dists[i:k-1]
	updatedDists := make([]float64, k)
	copy(updatedDists, left)
	updatedDists[i] = dist
	copy(updatedDists[i+1:], right)

	leftObjs, rightObjs := nearest[:i], nearest[i:k-1]
	updatedNearest := make([]Spatial, k)
	copy(updatedNearest, leftObjs)
	updatedNearest[i] = obj
	copy(updatedNearest[i+1:], rightObjs)

	return updatedDists, updatedNearest
}

func (tree *Rtree) nearestNeighbors(k int, p Point, n *node, dists []float64, nearest []Spatial) ([]Spatial, []float64) {
	if n.Leaf {
		for _, e := range n.Entries {
			dist := math.Sqrt(p.minDist(e.BoundingBox))
			dists, nearest = insertNearest(k, dists, nearest, dist, e.Obj)
		}
	} else {
		branches, branchDists := sortEntries(p, n.Entries)
		branches = pruneEntries(p, branches, branchDists)
		for _, e := range branches {
			nearest, dists = tree.nearestNeighbors(k, p, e.Child, dists, nearest)
		}
	}
	return nearest, dists
}
